import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter

import carla
from min_carla_env.env import CarlaEnv, CONFIG
from min_carla_env.recorder import EpisodeRecorder


class CNNEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 480, 480)
            conv_out = self.conv(dummy).shape[1]
        self.fc = nn.Linear(conv_out, feature_dim)
        self.output_dim = feature_dim

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x


class ActorCritic(nn.Module):
    def __init__(self, input_channels, feature_dim, action_dim, continuous):
        super().__init__()
        self.encoder = CNNEncoder(input_channels, feature_dim)
        self.continuous = continuous

        if continuous:
            self.actor_mean = nn.Linear(feature_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(feature_dim, action_dim)

        self.critic = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.encoder(x)
        value = self.critic(features)
        return features, value

    def get_action(self, x, deterministic=False):
        features, value = self.forward(x)
        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, x, action):
        features, value = self.forward(x)
        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
        return log_prob, value.squeeze(-1), entropy


class PPOBuffer:
    def __init__(self, obs_shape, action_dim, capacity, continuous):
        self.continuous = continuous
        obs_dtype = np.uint8
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        if continuous:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._capacity = capacity

    def store(self, obs, action, log_prob, value, reward, done):
        idx = self._ptr % self._capacity
        self.obs[idx] = obs
        if self.continuous:
            self.actions[idx] = action
        else:
            self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = done
        self._ptr += 1

    def finish_path(self, last_value=0.0):
        path_slice = slice(self._ptr - self._size(), self._ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        deltas = rewards[:-1] + 0.99 * values[1:] * (1 - self.dones[path_slice]) - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, 0.99 * 0.95)
        self.returns[path_slice] = self._discount_cumsum(rewards[:-1], 0.99)

    def _size(self):
        return self._ptr

    def get(self):
        size = self._size()
        adv = self.advantages[:size]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (
            self.obs[:size],
            self.actions[:size],
            self.log_probs[:size],
            adv,
            self.returns[:size],
        )

    def clear(self):
        self._ptr = 0

    @staticmethod
    def _discount_cumsum(x, discount):
        result = np.zeros_like(x, dtype=np.float32)
        cumsum = 0.0
        for i in reversed(range(len(x))):
            cumsum = x[i] + discount * cumsum
            result[i] = cumsum
        return result


class PPOTrainer:
    def __init__(
        self,
        env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64,
        buffer_capacity=2048,
        save_dir="checkpoints",
        log_dir="runs",
        record_interval=10,
        device=None,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.save_dir = save_dir
        self.record_interval = record_interval

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_shape = env.observation_space.shape
        if env.continuous:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n

        input_channels = 1 if len(obs_shape) == 2 else obs_shape[0]

        self.model = ActorCritic(
            input_channels=input_channels,
            feature_dim=256,
            action_dim=action_dim,
            continuous=env.continuous,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = PPOBuffer(
            obs_shape=obs_shape,
            action_dim=action_dim,
            capacity=buffer_capacity,
            continuous=env.continuous,
        )

        self.writer = SummaryWriter(log_dir=log_dir)
        self.recorder = EpisodeRecorder(save_dir=os.path.join(log_dir, "recordings"))

        os.makedirs(save_dir, exist_ok=True)

        self._total_steps = 0
        self._episode_count = 0
        self._best_reward = -float("inf")

    def _preprocess_obs(self, obs):
        if obs.ndim == 2:
            obs = obs[np.newaxis, ...]
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs = obs / 5.0
        return obs

    def collect_rollout(self, steps):
        obs = self.env.reset()
        ep_reward = 0.0
        ep_len = 0

        for _ in range(steps):
            obs_tensor = self._preprocess_obs(obs)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_tensor)

            if self.env.continuous:
                action_np = action.cpu().numpy().flatten()
            else:
                action_np = action.cpu().item()

            next_obs, reward, done, info = self.env.step(action_np)

            self.buffer.store(
                obs,
                action_np if self.env.continuous else action_np,
                log_prob.cpu().item(),
                value.cpu().item(),
                reward,
                float(done),
            )

            ep_reward += reward
            ep_len += 1
            self._total_steps += 1

            if done:
                self.buffer.finish_path(last_value=0.0)
                self._log_episode(ep_reward, ep_len, info)
                self._episode_count += 1

                if self._episode_count % self.record_interval == 0:
                    self.recorder.reset()
                    self._record_episode()

                obs = self.env.reset()
                ep_reward = 0.0
                ep_len = 0
            else:
                obs = next_obs

        if ep_len > 0:
            obs_tensor = self._preprocess_obs(obs)
            with torch.no_grad():
                _, _, last_value = self.model.get_action(obs_tensor)
            self.buffer.finish_path(last_value=last_value.cpu().item())

    def _record_episode(self):
        obs = self.env.reset()
        done = False
        while not done:
            obs_tensor = self._preprocess_obs(obs)
            with torch.no_grad():
                action, _, _ = self.model.get_action(obs_tensor, deterministic=True)
            if self.env.continuous:
                action_np = action.cpu().numpy().flatten()
            else:
                action_np = action.cpu().item()
            next_obs, reward, done, info = self.env.step(action_np)
            self.recorder.record_step(obs, action_np, reward, info)
            obs = next_obs
        self.recorder.save(episode_id=self._episode_count)

    def _log_episode(self, ep_reward, ep_len, info):
        self.writer.add_scalar("episode/reward", ep_reward, self._episode_count)
        self.writer.add_scalar("episode/length", ep_len, self._episode_count)
        self.writer.add_scalar("episode/total_steps", self._total_steps, self._episode_count)
        if "kmh" in info:
            self.writer.add_scalar("episode/kmh", info["kmh"], self._episode_count)
        if "reward_breakdown" in info:
            for k, v in info["reward_breakdown"].items():
                self.writer.add_scalar(f"reward/{k}", v, self._episode_count)

    def update(self):
        obs, actions, old_log_probs, advantages, returns = self.buffer.get()

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 5.0
        if self.env.continuous:
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                log_probs, values, entropy = self.model.evaluate(batch_obs, batch_actions)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()

        avg_policy_loss = total_policy_loss / max(n_updates, 1)
        avg_value_loss = total_value_loss / max(n_updates, 1)
        avg_entropy = total_entropy / max(n_updates, 1)

        self.writer.add_scalar("train/policy_loss", avg_policy_loss, self._total_steps)
        self.writer.add_scalar("train/value_loss", avg_value_loss, self._total_steps)
        self.writer.add_scalar("train/entropy", avg_entropy, self._total_steps)

        return avg_policy_loss, avg_value_loss, avg_entropy

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.save_dir, f"model_{self._total_steps}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
            "episode_count": self._episode_count,
        }, path)
        print(f"模型已保存: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._total_steps = checkpoint["total_steps"]
        self._episode_count = checkpoint["episode_count"]
        print(f"模型已加载: {path} (step={self._total_steps})")

    def train(self, total_steps, rollout_steps=None, save_interval=50000):
        if rollout_steps is None:
            rollout_steps = self.buffer_capacity

        print(f"开始训练: total_steps={total_steps}, device={self.device}")
        print(f"动作空间: {'连续' if self.env.continuous else '离散'}")

        start_time = time.time()
        while self._total_steps < total_steps:
            self.collect_rollout(rollout_steps)
            p_loss, v_loss, ent = self.update()

            if self._episode_count > 0 and self._episode_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = self._total_steps / max(elapsed, 1)
                print(
                    f"Episode {self._episode_count:5d} | "
                    f"Steps {self._total_steps:8d} | "
                    f"FPS {fps:.1f} | "
                    f"P_Loss {p_loss:.4f} | "
                    f"V_Loss {v_loss:.4f} | "
                    f"Entropy {ent:.4f}"
                )

            if self._total_steps % save_interval < rollout_steps:
                self.save()

        self.save(os.path.join(self.save_dir, "model_final.pt"))
        self.writer.close()
        print("训练完成!")

    def evaluate(self, num_episodes=5, render=False):
        print(f"\n开始评估 ({num_episodes} episodes)...")
        total_rewards = []
        total_lengths = []

        for ep in range(num_episodes):
            obs = self.env.reset()
            ep_reward = 0.0
            ep_len = 0
            done = False

            while not done:
                obs_tensor = self._preprocess_obs(obs)
                with torch.no_grad():
                    action, _, _ = self.model.get_action(obs_tensor, deterministic=True)
                if self.env.continuous:
                    action_np = action.cpu().numpy().flatten()
                else:
                    action_np = action.cpu().item()
                obs, reward, done, info = self.env.step(action_np)
                ep_reward += reward
                ep_len += 1

            total_rewards.append(ep_reward)
            total_lengths.append(ep_len)
            print(f"  Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_length = np.mean(total_lengths)
        print(f"\n评估结果: reward={avg_reward:.2f} ± {std_reward:.2f}, avg_length={avg_length:.1f}")
        return avg_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description="Carla 2D DeepRL PPO Training")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--continuous", action="store_true", default=True)
    parser.add_argument("--discrete", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    continuous = args.continuous and not args.discrete

    config = CONFIG.copy()
    config["continuous"] = continuous
    config["render"] = args.render

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    env = CarlaEnv(
        client,
        config,
        world_config={
            "render": args.render,
            "fast": args.fast,
            "town": args.town,
        },
    )

    trainer = PPOTrainer(
        env=env,
        lr=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    if args.load:
        trainer.load(args.load)

    if args.evaluate:
        trainer.evaluate(num_episodes=args.eval_episodes)
    else:
        try:
            trainer.train(total_steps=args.total_steps)
        except KeyboardInterrupt:
            print("\n训练被中断，保存模型...")
            trainer.save(os.path.join(args.save_dir, "model_interrupted.pt"))
        finally:
            env.close()


if __name__ == "__main__":
    main()