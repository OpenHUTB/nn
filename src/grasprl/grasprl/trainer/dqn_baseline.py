import torch.nn as nn
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import math
import random
import copy
import os
import sys
import cv2

sys.path.append(r"D:\nn\src\grasp_rl\grasp_rl")

from envs.grasp import GraspRobot
from modules.ddpg import ReplayBuffer, Transition
from modules.qnet import MULTIDISCRETE_RESNET

from tqdm import tqdm
import cv2

MAX_POSSIBLE_SAMPLES = 12
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
        x = torch.clamp(x, -1.0, 1.0)
        return x

class DQN_Trainer(object):
    def __init__(
        self,
        learning_rate=0.001,
        mem_size=2000,
        eps_start=1.0,
        eps_end=0.,
        eps_decay=200,
        seed=20,
        log_dir="test",
        render_mode="rgb"):

        self.writer = SummaryWriter(f"grasprl/log/DQN/{log_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = GraspRobot(render_mode=render_mode)
        self.memory = ReplayBuffer(mem_size, simple=False)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        self.q_net = MULTIDISCRETE_RESNET(1).to(self.device)
        self.feat_enhance = VisualFeatureEnhancer().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=0.00002)
        self.criterion = torch.nn.SmoothL1Loss(reduction="none").to(device=self.device)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
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
        pixel_y = max_idx // self.env.IMAGE_HEIGHT
        pixel_x = max(0, min(pixel_x, self.env.IMAGE_WIDTH - 1))
        pixel_y = max(0, min(pixel_y, self.env.IMAGE_HEIGHT - 1))
    
        if 0 <= pixel_x < self.env.IMAGE_WIDTH and 0 <= pixel_y < self.env.IMAGE_HEIGHT:
            depth = self.depth_before[pixel_x][pixel_y]
        else:
            depth = 0.0
        action = self.env.pixel2world(1, pixel_x, pixel_y, depth)
        return action

    def limit_action(self, action):
        action = np.clip(action, [-0.25, -0.25, self.env.TABLE_HEIGHT + 0.05], [0.25, 0.25, 2])
        return list(action)

    def select_action_by_eps_random(self, state):
        eps_threshold = self._get_eps_threshold()
        if random.random() > eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                q_max = self.q_net(state).argmax()
                q_max = torch.tensor([[q_max]], dtype=torch.long)
                return q_max
        else:
            self.last_action = "random"
            action = np.random.randint(low=0, high=self.env.IMAGE_WIDTH * self.env.IMAGE_HEIGHT)
            return torch.tensor([[action]], dtype=torch.long)

    def select_action_by_eps(self, state):
        eps_threshold = self._get_eps_threshold()
        if random.random() > eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                q_max = self.q_net(state).argmax()
                q_max = torch.tensor([[q_max]], dtype=torch.long)
                return q_max
        else:
            self.last_action = "random"
            r = g = b = 0
            threshold = 0
            while r == g and g == b:

                action = np.random.randint(low=0, high=(self.env.IMAGE_WIDTH - 1) * (self.env.IMAGE_HEIGHT - 1))
                pixel_x = action % self.env.IMAGE_WIDTH
                pixel_y = action // self.env.IMAGE_WIDTH
                r, g, b = self.env.observation["rgb"][pixel_x][pixel_y]
                threshold += 1
                if threshold == 10:
                    break
            return torch.tensor([[action]], dtype=torch.long)
    
    def _get_eps_threshold(self):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        self.writer.add_scalar("Epslion", eps_threshold, self.steps_done)
        return eps_threshold
    
    def select_action_by_instruction(self, state):
        eps_threshold = self._get_eps_threshold()
        if random.random() > eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                q_max = self.q_net(state).argmax()
                q_max = torch.tensor([[q_max]], dtype=torch.long)
                return q_max
        else:
            self.last_action = "instruction"
            action = None
            for obj_name in self.env.target_objects:
                wx, wy, wz = self.env.get_body_com(obj_name)
                if -0.224 <= wx <= 0.224 and -0.224 <= wy <= 0.224 and wz >= 0.9:

                    px, py = self.env.world2pixel(cam_id=1, x=wx, y=wy, z=wz)
                    action = py * self.env.IMAGE_WIDTH + px
                    action = max(0, min(action, self.env.IMAGE_WIDTH * self.env.IMAGE_HEIGHT - 1))
                    break
            if action is None:
                action = np.random.randint(low=0, high=self.env.IMAGE_WIDTH * self.env.IMAGE_HEIGHT)
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
    
        self.optimizer.zero_grad()
    
        
        q_out = self.q_net(state_batch).view(-1, self.env.IMAGE_WIDTH * self.env.IMAGE_HEIGHT)
        q_pred = q_out.gather(1, action_batch)
        with torch.no_grad():
            q_next = self.q_net(next_state_batch).max(1, keepdim=True)[0]
            q_target = reward_batch + gamma * q_next
    
        loss = self.criterion(q_pred, q_target).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar("losses", loss.item(), self.steps_done)

    def predict(self, state, show=False):
        with torch.no_grad():
            affordance = self.q_net(state)
            q_max = affordance.argmax()
            action = self.transform_action(q_max)
            action = self.limit_action(action)
            if show:
                cv2.imshow("affordance", affordance.cpu().numpy())
                cv2.waitKey(0)
        return action

    def save(self, path_name, filename):
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        torch.save(self.q_net.state_dict(), path_name + filename + "_qnet")

    def load(self, filename):
        self.q_net.load_state_dict(torch.load(f=filename + "_qnet", map_location=self.device))

    def count(self, greedy, random_num):
        eps_threshold = self._get_eps_threshold()
        if random.random() > eps_threshold:
            greedy += 1
        else:
            self.last_action = "random"
            random_num += 1
        return greedy, random_num

    def save_dataset_sample(self, action, reward, info, iter_num):
        data_dir = "grasprl/dataset/grasp_samples"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
        rgb_img = self.env.observation["rgb"]
        rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{data_dir}/rgb_{iter_num}.png", rgb_img_bgr)
    
        depth_img = self.env.observation["depth"]
        np.save(f"{data_dir}/depth_{iter_num}.npy", depth_img)
    
        label = {
            "action": action,
            "grasp_success": 1 if info["grasp"] == "Success" else 0,
            "reward": reward,
            "iter_num": iter_num
        }
        np.save(f"{data_dir}/label_{iter_num}.npy", label)

def main():
    max_iter = 50
    grasp_success = 0
    loop = tqdm(range(1, max_iter + 1))

    trainer = DQN_Trainer(log_dir="resnet_dqn_insne",render_mode="human")

    state = trainer.env.reset_without_random()
    state = trainer.transform_state(state)
    for i_iter in loop:
        max_idx = trainer.select_action_by_instruction(state)
        action = trainer.transform_action(max_idx)
        action = trainer.limit_action(action)
        next_state, reward, done, info = trainer.env.step(action)
        import time
        if trainer.env.render_mode == "human":
            time.sleep(0.05)
        loop.set_description(f"iter [{i_iter}]/[{max_iter}]")
        loop.set_postfix(grasp_info=info['grasp'], reward=reward, action=trainer.last_action)
        if info["grasp"] == "Success":
            grasp_success += 1
        if done:
            trainer.env.reset_without_random()
        success_rate = grasp_success / i_iter if i_iter > 0 else 0.0
        trainer.writer.add_scalar("Grasping performance(Success rate)", success_rate, trainer.steps_done)
        reward = torch.tensor([[reward]], dtype=torch.float32)
        next_state = trainer.transform_state(next_state)
        trainer.memory.push(state.detach(), max_idx, next_state.detach(), reward)
        state = next_state
        trainer.save_dataset_sample(action, reward.item(), info, i_iter)
        trainer.learn()

    trainer.save(path_name="grasprl/trained/resnet/resnet", filename="insne")
    trainer.writer.close()


if __name__ == "__main__":
    main()