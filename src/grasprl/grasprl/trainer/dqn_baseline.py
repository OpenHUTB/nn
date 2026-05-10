import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import math
import random
import os
import sys
import cv2
#改成相对路径 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.grasp import GraspRobot
from modules.ddpg import ReplayBuffer, Transition
from modules.qnet import MULTIDISCRETE_RESNET
from tqdm import tqdm

MAX_POSSIBLE_SAMPLES = 32
NUMBER_ACCUMULATIONS_BEFORE_UPDATE = 1
BATCH_SIZE = MAX_POSSIBLE_SAMPLES * NUMBER_ACCUMULATIONS_BEFORE_UPDATE

class VisualFeatureEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.clamp(x, -1.0, 1.0)

class DQN_Trainer(object):
    def __init__(self, learning_rate=0.0005, mem_size=10000, eps_start=1.0, eps_end=0.01,
                 eps_decay=8000, seed=20, log_dir="test", render_mode=None):
        self.writer = SummaryWriter(f"grasprl/log/DQN/{log_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = GraspRobot(render_mode=render_mode)
        self.memory = ReplayBuffer(mem_size, simple=False)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.best_success_rate = 0.0

        self.q_net = MULTIDISCRETE_RESNET(1).to(self.device)
        self.target_q_net = MULTIDISCRETE_RESNET(1).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        
        self.feat_enhance = VisualFeatureEnhancer().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=0.0001)
        self.criterion = torch.nn.SmoothL1Loss(reduction="none").to(self.device)
        self.target_update_freq = 100

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def transform_state(self, state):
        self.depth_before = state["depth"]
        depth_img = np.asarray(state["depth"])
        depth_img = depth_img.max() - depth_img
        img_trans = T.ToTensor()
        img_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor_rgb = img_trans(state["rgb"]).unsqueeze(0)
        tensor_rgb = img_normalize(tensor_rgb)
        tensor_depth = img_trans(depth_img).unsqueeze(0)
        tensor_obs = torch.cat([tensor_rgb, tensor_depth], 1)
        tensor_obs = self.feat_enhance(tensor_obs)
        return tensor_obs.to(self.device)

    def transform_action(self, max_idx):
        max_idx = max_idx.item()
        pixel_x = max_idx % self.env.IMAGE_WIDTH
        pixel_y = max_idx // self.env.IMAGE_WIDTH
        pixel_x = max(0, min(pixel_x, self.env.IMAGE_WIDTH - 1))
        pixel_y = max(0, min(pixel_y, self.env.IMAGE_HEIGHT - 1))
        if 0 <= pixel_x < self.env.IMAGE_WIDTH and 0 <= pixel_y < self.env.IMAGE_HEIGHT:
            depth = self.depth_before[pixel_y][pixel_x]
        else:
            depth = 0.0
        action = self.env.pixel2world(1, pixel_x, pixel_y, depth)
        return action

    def limit_action(self, action):
        return list(np.clip(action, [-0.25, -0.25, self.env.TABLE_HEIGHT + 0.05], [0.25, 0.25, 2]))

    def _get_eps_threshold(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        self.writer.add_scalar("Epslion", eps, self.steps_done)
        return eps

    def select_action_by_instruction(self, state):
        if random.random() > self._get_eps_threshold():
            self.last_action = "greedy"
            with torch.no_grad():
                q_max = self.q_net(state).argmax()
                return torch.tensor([[q_max]], dtype=torch.long)
        else:
            self.last_action = "instruction"
            action = None
            for obj_name in self.env.target_objects:
                wx, wy, wz = self.env.get_body_com(obj_name)
                if -0.224 <= wx <= 0.224 and -0.224 <= wy <= 0.224 and wz >= 0.9:
                    px, py = self.env.world2pixel(1, wx, wy, wz)
                    old_w, old_h = 224, 224
                    new_w, new_h = self.env.IMAGE_WIDTH, self.env.IMAGE_HEIGHT
                    px = int(px * new_w / old_w)
                    py = int(py * new_h / old_h)
                    px = max(0, min(px, new_w - 1))
                    py = max(0, min(py, new_h - 1))
                    action = py * new_w + px
                    break
            if action is None:
                action = np.random.randint(0, self.env.IMAGE_WIDTH * self.env.IMAGE_HEIGHT)
            return torch.tensor([[action]], dtype=torch.long)

    def learn(self, gamma=0.99):
        if len(self.memory) < BATCH_SIZE:
            print("Filling the replay buffer ...")
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        reward_batch = reward_batch.float().view(-1, 1)
        q_out = self.q_net(state_batch).view(-1, self.env.IMAGE_WIDTH * self.env.IMAGE_HEIGHT)
        q_pred = q_out.gather(1, action_batch)
        
        with torch.no_grad():
            q_next = self.target_q_net(next_state_batch).max(1, keepdim=True)[0]
            q_target = reward_batch + gamma * q_next

        loss = self.criterion(q_pred, q_target).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.writer.add_scalar("losses", loss.item(), self.steps_done)

        if self.steps_done % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path_name, filename):
        os.makedirs(path_name, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(path_name, filename + "_qnet"))
        torch.save(self.target_q_net.state_dict(), os.path.join(path_name, filename + "_target_qnet"))
        torch.save(self.feat_enhance.state_dict(), os.path.join(path_name, filename + "_feat_enhance"))
        
    def save_dataset_sample(self, action, reward, info, iter_num):
        data_dir = "grasprl/dataset/grasp_samples"
        os.makedirs(data_dir, exist_ok=True)
        rgb = self.env.observation["rgb"]
        cv2.imwrite(f"{data_dir}/rgb_{iter_num}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(f"{data_dir}/depth_{iter_num}.npy", self.env.observation["depth"])
        label = {"action": action, "grasp_success": 1 if info["grasp"] == "Success" else 0,
                 "reward": reward, "iter_num": iter_num}
        np.save(f"{data_dir}/label_{iter_num}.npy", label)

def main():
    max_iter = 100
    trainer = DQN_Trainer(log_dir="resnet_dqn_insne_v2", render_mode="human")
    state = trainer.env.reset_without_random()
    state = trainer.transform_state(state)
    loop = tqdm(range(1, max_iter + 1))
    grasp_success = 0
    recent_successes = []

    for i_iter in loop:
        max_idx = trainer.select_action_by_instruction(state)
        action = trainer.transform_action(max_idx)
        action = trainer.limit_action(action)
        next_state, reward, done, info = trainer.env.step(action)
        loop.set_description(f"iter [{i_iter}]/[{max_iter}]")
        loop.set_postfix(grasp_info=info['grasp'], reward=reward, action=trainer.last_action)
        
        if info["grasp"] == "Success":
            grasp_success += 1
            recent_successes.append(1)
        else:
            recent_successes.append(0)
        
        if len(recent_successes) > 50:
            recent_successes.pop(0)
        
        if done:
            state = trainer.env.reset_without_random()
            state = trainer.transform_state(state)
        else:
            success_rate = grasp_success / i_iter
            recent_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
            
            trainer.writer.add_scalar("Grasping performance(Success rate)", success_rate, trainer.steps_done)
            trainer.writer.add_scalar("Recent Success Rate (last 50)", recent_success_rate, trainer.steps_done)
            
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
            next_state = trainer.transform_state(next_state)
            trainer.memory.push(state.detach(), max_idx, next_state.detach(), reward_tensor)
            state = next_state
            trainer.save_dataset_sample(action, reward, info, i_iter)
            trainer.learn()

            if recent_success_rate > trainer.best_success_rate and i_iter > 100:
                trainer.best_success_rate = recent_success_rate
                trainer.save("grasprl/trained/resnet/resnet_best", f"insne_{i_iter}")

    trainer.save("grasprl/trained/resnet/resnet", "insne_final")
    trainer.writer.close()

if __name__ == "__main__":
    main()


