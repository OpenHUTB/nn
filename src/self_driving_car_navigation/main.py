import torch
import torch.nn as nn
import argparse 
from torch.optim import Adam
from utils.dataloader import get_dataloader
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from models.sagm import SelfAssessmentGradientModel  

class IntegratedSystem(nn.Module):
    
    def __init__(self, device='cpu', state_dim=128, action_dim=2):
        super().__init__()
        self.device = device
        self.perception = PerceptionModule().to(self.device)
        self.attention = CrossDomainAttention(num_blocks=2,embed_dim=64).to(self.device)
        self.decision = DecisionModule().to(self.device)
        self.sagm = SelfAssessmentGradientModel(hidden_dim=64 ).to(self.device)  

    def forward(self, image, lidar_data, imu_data, action):
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        
        policy, value = self.decision(fused_features)
        
        action_3d = action.unsqueeze(1)
        seq_len = fused_features.shape[1]
        action_3d = action_3d.repeat(1, seq_len, 1)

        sagm_q_value = self.sagm(fused_features, action_3d)  
        return policy, value, sagm_q_value

def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (image, lidar_data, imu_data, target_action) in enumerate(dataloader):
            image, lidar_data, imu_data, target_action = image.to(device), lidar_data.to(device), imu_data.to(device), target_action.to(device)
            optimizer.zero_grad()
            policy_output, value_output, sagm_q_value = model(image, lidar_data, imu_data, target_action)
            
            loss = (nn.MSELoss()(policy_output, target_action) +
                    nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True)) +
                    nn.MSELoss()(sagm_q_value, target_action.sum(dim=1, keepdim=True)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
    print('Training complete')

def test_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for image, lidar_data, imu_data, target_action in dataloader:
            image, lidar_data, imu_data, target_action = image.to(device), lidar_data.to(device), imu_data.to(device), target_action.to(device)
            policy_output, value_output, sagm_q_value = model(image, lidar_data, imu_data, target_action)
            loss = (nn.MSELoss()(policy_output, target_action) +
                    nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True)) +
                    nn.MSELoss()(sagm_q_value, target_action.sum(dim=1, keepdim=True)))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Test Average Loss: {avg_loss:.4f}')
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IntegratedSystem(device=device)
    optimizer = Adam(model.parameters(), lr=0.001)
    dataloader = get_dataloader()

    if args.mode == "train":
        train_model(model, dataloader, optimizer, device)
    elif args.mode == "test":
        test_model(model, dataloader, device)