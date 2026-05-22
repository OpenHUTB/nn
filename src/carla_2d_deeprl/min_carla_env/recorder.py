import os
import json
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EpisodeRecorder:
    def __init__(self, save_dir="recordings", compress=True):
        self.save_dir = save_dir
        self.compress = compress
        self._obs_list = []
        self._action_list = []
        self._reward_list = []
        self._info_list = []
        self._done = False
        self._total_reward = 0.0
        self._step_count = 0
        os.makedirs(save_dir, exist_ok=True)

    def reset(self):
        self._obs_list = []
        self._action_list = []
        self._reward_list = []
        self._info_list = []
        self._done = False
        self._total_reward = 0.0
        self._step_count = 0

    def record_step(self, obs, action, reward, info):
        self._obs_list.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
        self._action_list.append(action)
        self._reward_list.append(reward)
        self._info_list.append(info)
        self._total_reward += reward
        self._step_count += 1

    def save(self, episode_id=None, metadata=None):
        if episode_id is None:
            episode_id = int(time.time())

        save_path = os.path.join(self.save_dir, f"episode_{episode_id}")
        os.makedirs(save_path, exist_ok=True)

        obs_array = np.array(self._obs_list, dtype=np.uint8)
        action_array = np.array(self._action_list, dtype=np.float32)
        reward_array = np.array(self._reward_list, dtype=np.float32)

        if self.compress:
            np.savez_compressed(
                os.path.join(save_path, "data.npz"),
                observations=obs_array,
                actions=action_array,
                rewards=reward_array,
            )
        else:
            np.savez(
                os.path.join(save_path, "data.npz"),
                observations=obs_array,
                actions=action_array,
                rewards=reward_array,
            )

        summary = {
            "episode_id": episode_id,
            "total_reward": float(self._total_reward),
            "step_count": self._step_count,
            "metadata": metadata or {},
        }
        with open(os.path.join(save_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Episode {episode_id} 已保存: {self._step_count} 步, "
            f"总奖励={self._total_reward:.2f}, 路径={save_path}"
        )
        return save_path

    @staticmethod
    def load(episode_path):
        data = np.load(os.path.join(episode_path, "data.npz"))
        with open(os.path.join(episode_path, "summary.json"), "r") as f:
            summary = json.load(f)
        return {
            "observations": data["observations"],
            "actions": data["actions"],
            "rewards": data["rewards"],
            "summary": summary,
        }

    @staticmethod
    def list_episodes(save_dir="recordings"):
        if not os.path.exists(save_dir):
            return []
        episodes = []
        for name in os.listdir(save_dir):
            ep_path = os.path.join(save_dir, name)
            summary_path = os.path.join(ep_path, "summary.json")
            if os.path.isdir(ep_path) and os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                episodes.append(summary)
        return sorted(episodes, key=lambda x: x["episode_id"])


class EpisodeReplayer:
    def __init__(self, episode_path):
        self.data = EpisodeRecorder.load(episode_path)
        self._idx = 0
        self._length = len(self.data["actions"])

    def __len__(self):
        return self._length

    def reset(self):
        self._idx = 0

    def step(self):
        if self._idx >= self._length:
            return None, None, None, True
        obs = self.data["observations"][self._idx]
        action = self.data["actions"][self._idx]
        reward = self.data["rewards"][self._idx]
        done = (self._idx == self._length - 1)
        self._idx += 1
        return obs, action, reward, done

    def render_frame(self, idx=None):
        import cv2
        if idx is None:
            idx = self._idx
        if idx >= self._length:
            return None
        obs = self.data["observations"][idx]
        if obs.ndim == 2:
            obs = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return obs