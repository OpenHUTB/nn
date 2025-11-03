import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

class SelfAssessmentGradientModel(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1):
        super(SelfAssessmentGradientModel, self).__init__()
        self.fc1 = nn.Linear(66, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)  
        x = x.reshape(-1, 66)
        x = self.activation(self.fc1(x))
        q_value = self.fc2(x)
        q_value = q_value.view(state.size(0), state.size(1), -1)
        return q_value
if __name__ == "__main__":
    state_dim = 10  
    action_dim = 2  
    input_dim = state_dim + action_dim
    hidden_dim = 64
    output_dim = 1
    discount_factor = 0.99
    weight_sagm = 0.5  

    sagm = SelfAssessmentGradientModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    critic = nn.Linear(input_dim, 1)  
    actor = nn.Linear(state_dim, action_dim)  

    optimizer_sagm = optim.Adam(sagm.parameters(), lr=0.0001)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.0001)
    optimizer_actor = optim.Adam(actor.parameters(), lr=0.0001)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log_interval = 10  

    def update_sagm(sagm, optimizer_sagm, state, action, target_q_value):
        sagm_q_value = sagm(state, action)
        sagm_loss = F.mse_loss(sagm_q_value, target_q_value)
        optimizer_sagm.zero_grad()
        sagm_loss.backward(retain_graph=True)
        optimizer_sagm.step()
        return sagm_q_value

    def compute_target_q_value(reward, next_state, critic, actor):
        with torch.no_grad():
            next_action = actor(next_state)
            target_q_value = reward + discount_factor * critic(torch.cat((next_state, next_action), dim=-1))
        return target_q_value

    def update_actor_critic(critic, actor, sagm_q_value, state, action):
        total_q_value = (1 - weight_sagm) * critic(torch.cat((state, action), dim=-1)) + weight_sagm * sagm_q_value
        total_loss = -total_q_value.mean()  
        
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        total_loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()

    total_episodes = 1000  
    for episode in range(total_episodes):
        state = torch.randn(1, state_dim)  
        done = False
        episode_reward = 0

        while not done:
            action = actor(state)
            next_state = torch.randn(1, state_dim)  
            reward = torch.tensor([[1.0]])  
            done = torch.rand(1).item() > 0.95  
            episode_reward += reward.item()
            
            target_q_value = compute_target_q_value(reward, next_state, critic, actor)
            
            sagm_q_value = update_sagm(sagm, optimizer_sagm, state, action, target_q_value)
            
            update_actor_critic(critic, actor, sagm_q_value, state, action)
            
            state = next_state

        if episode % log_interval == 0:
            logging.info(f'Episode: {episode}, Reward: {episode_reward:.2f}')
