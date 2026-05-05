import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs.grasp import GraspRobot
from modules.ddpg import ReplayBuffer, Transition
from modules.qnet import MULTIDISCRETE_RESNET

BATCH_SIZE = 16
GAMMA = 0.95
LR = 0.0005

class VisualFeatureEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,4,3,padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        return torch.clamp(x,-1,1)

class DQN_Trainer:
    def __init__(self, lr=LR, mem_size=10000, eps_start=0.9, eps_end=0.05, eps_decay=5000,
                 seed=42, log_dir="test", render_mode=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = GraspRobot(render_mode=render_mode)
        self.memory = ReplayBuffer(mem_size)
        self.writer = SummaryWriter(f"grasprl/log/DQN/{log_dir}")
        self.eps_start,self.eps_end,self.eps_decay = eps_start, eps_end, eps_decay
        self.steps_done = 0

        self.q_net = MULTIDISCRETE_RESNET(1).to(self.device,dtype=torch.float32)
        self.feat_enhance = VisualFeatureEnhancer().to(self.device,dtype=torch.float32)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, eps=1e-4)
        self.criterion = nn.SmoothL1Loss(reduction="mean").to(self.device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def transform_state(self,state):
        depth = np.max(state["depth"]) - state["depth"]
        depth = (depth - np.min(depth)) / (np.max(depth)-np.min(depth)+1e-8)
        depth = depth.astype(np.float32)

        rgb = state["rgb"].astype(np.float32)/255.0
        tensor_rgb = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device,dtype=torch.float32)
        tensor_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device,dtype=torch.float32)

        tensor_obs = torch.cat([tensor_rgb, tensor_depth], dim=1).float()
        with torch.no_grad():
            tensor_obs = self.feat_enhance(tensor_obs)
        return tensor_obs

    def _get_eps(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.*self.steps_done/self.eps_decay)
        self.steps_done += 1
        if self.steps_done % 100 == 0:
            self.writer.add_scalar("Epsilon",eps,self.steps_done)
        return eps

    def select_action(self,state):
        eps = self._get_eps()
        if random.random() > eps:
            with torch.no_grad():
                q_vals = self.q_net(state).view(-1)   # flatten H*W
                max_idx = torch.argmax(q_vals)
                return max_idx.unsqueeze(0).unsqueeze(0)  # shape [1,1]
        else:
            valid_idx = list(range(self.env.IMAGE_WIDTH*self.env.IMAGE_HEIGHT))
            return torch.tensor([[random.choice(valid_idx)]],dtype=torch.long,device=self.device)

    def transform_action(self,max_idx,depth_before):
        idx = max_idx.item()
        px = idx % self.env.IMAGE_WIDTH
        py = idx // self.env.IMAGE_WIDTH
        px = np.clip(px,0,self.env.IMAGE_WIDTH-1)
        py = np.clip(py,0,self.env.IMAGE_HEIGHT-1)
        depth = depth_before[py][px] if depth_before[py][px]>0 else np.mean(depth_before)
        return self.env.pixel2world(1,px,py,depth)

    def limit_action(self,action):
        return np.clip(action,
                       [-0.25,-0.25,self.env.TABLE_HEIGHT+0.05],
                       [0.25,0.25,2.0]).astype(np.float32).tolist()

    def learn(self):
        if len(self.memory)<BATCH_SIZE: return
        trans = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*trans))

        s = torch.cat(batch.state).to(self.device,dtype=torch.float32)
        a = torch.cat(batch.action).to(self.device,dtype=torch.long)
        r = torch.cat(batch.reward).view(-1,1).float().to(self.device)
        ns = torch.cat(batch.next_state).to(self.device,dtype=torch.float32)

        B = s.size(0)
        # flatten network输出
        q_s = self.q_net(s).view(B,-1)
        q_ns = self.q_net(ns).view(B,-1)

        # gather idx对应动作
        q_pred = q_s.gather(1,a)             # [B,1]
        with torch.no_grad():
            q_next = q_ns.max(1,keepdim=True)[0]   # [B,1]
            q_target = r + GAMMA*q_next             # [B,1]

        loss = self.criterion(q_pred,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),1.0)
        self.optimizer.step()

        if self.steps_done%100==0:
            self.writer.add_scalar("loss",loss.item(),self.steps_done)

def main():
    trainer = DQN_Trainer(log_dir="resnet_dqn_opt", render_mode=None)
    state = trainer.env.reset_without_random()
    state = trainer.transform_state(state)
    grasp_success = 0
    total_reward = 0
    loop = tqdm(range(1,501),desc="Training DQN")
    for i in loop:
        idx = trainer.select_action(state)
        action = trainer.transform_action(idx,trainer.env.observation["depth"])
        action = trainer.limit_action(action)
        next_state,reward,done,info = trainer.env.step(action)
        total_reward += reward

        trainer.memory.push(
            state,
            idx,
            torch.tensor([[reward]],dtype=torch.float32,device=trainer.device),
            trainer.transform_state(next_state)
        )

        trainer.learn()
        loop.set_postfix(grasp_info=info["grasp"], reward=round(reward,2), total_reward=round(total_reward,2), success_num=grasp_success)

        if info["grasp"]=="Success":
            grasp_success +=1
        state = trainer.transform_state(next_state) if not done else trainer.transform_state(trainer.env.reset_without_random())

    print(f"\n训练完成 | 总成功次数: {grasp_success} | 成功率: {grasp_success/500:.2%}")
    trainer.writer.close()

if __name__=="__main__":
    main()