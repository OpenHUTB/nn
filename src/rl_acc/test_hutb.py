import numpy as np
import matplotlib.pyplot as plt
from acc_env import HutbACCEnv
from config import TEST_EPISODES, TEST_RENDER

def test_hutb_env():
    env = HutbACCEnv(render_mode='human' if TEST_RENDER else None)
    
    all_rewards = []
    all_ego_speeds = []
    all_distances = []
    
    for episode in range(TEST_EPISODES):
        obs, _ = env.reset()
        episode_rewards = []
        ego_speeds = []
        distances = []
        
        terminated = False
        while not terminated:
            action = np.array([0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            ego_speeds.append(info['ego_speed'])
            distances.append(info['distance'])
            
            if TEST_RENDER:
                env.render()
        
        all_rewards.append(np.sum(episode_rewards))
        all_ego_speeds.append(ego_speeds)
        all_distances.append(distances)
        
        print(f"Episode {episode+1}: Total Reward = {np.sum(episode_rewards):.2f}")
    
    env.close()
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for i, speeds in enumerate(all_ego_speeds):
        plt.plot(speeds, label=f"Episode {i+1}")
    plt.title('Ego Vehicle Speed')
    plt.xlabel('Step')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i, dists in enumerate(all_distances):
        plt.plot(dists, label=f"Episode {i+1}")
    plt.title('Distance to Front Vehicle')
    plt.xlabel('Step')
    plt.ylabel('Distance (m)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hutb_test_results.png')
    print("\nTest completed! Results saved to hutb_test_results.png")
    
    print("\n=== Test Summary ===")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Max Reward: {np.max(all_rewards):.2f}")
    print(f"Min Reward: {np.min(all_rewards):.2f}")

if __name__ == "__main__":
    test_hutb_env()
