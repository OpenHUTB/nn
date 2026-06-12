import numpy as np
import matplotlib.pyplot as plt
from reward_functions import RewardFunction

def test_reward_functions():
    reward_fn = RewardFunction()
    
    speeds = np.linspace(0, 35, 100)
    distances = np.linspace(0, 100, 100)
    accelerations = np.linspace(-3, 2, 100)
    
    speed_rewards = []
    for speed in speeds:
        _, rewards = reward_fn.calculate(speed, 25, 30, 0)
        speed_rewards.append(rewards['speed'])
    
    distance_rewards = []
    for dist in distances:
        _, rewards = reward_fn.calculate(25, 25, dist, 0)
        distance_rewards.append(rewards['distance'])
    
    comfort_rewards = []
    for accel in accelerations:
        _, rewards = reward_fn.calculate(25, 25, 30, accel)
        comfort_rewards.append(rewards['comfort'])
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(speeds, speed_rewards)
    plt.title('Speed Tracking Reward')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Reward')
    plt.axvline(x=25, color='r', linestyle='--', label='Target Speed')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(distances, distance_rewards)
    plt.title('Distance Control Reward')
    plt.xlabel('Distance (m)')
    plt.ylabel('Reward')
    plt.axvline(x=15, color='r', linestyle='--', label='Min Safe Distance')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(accelerations, comfort_rewards)
    plt.title('Comfort Reward')
    plt.xlabel('Acceleration (m/s²)')
    plt.ylabel('Reward')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Acceleration')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reward_function_analysis.png')
    print("Reward function analysis saved to reward_function_analysis.png")
    
    print("\n=== Reward Function Test ===")
    test_cases = [
        (25, 25, 30, 0),
        (20, 25, 20, 1),
        (30, 20, 10, -2),
    ]
    
    for i, (ego_speed, front_speed, distance, acceleration) in enumerate(test_cases):
        total_reward, rewards = reward_fn.calculate(ego_speed, front_speed, distance, acceleration)
        print(f"\nTest Case {i+1}:")
        print(f"  Ego Speed: {ego_speed} m/s")
        print(f"  Front Speed: {front_speed} m/s")
        print(f"  Distance: {distance} m")
        print(f"  Acceleration: {acceleration} m/s²")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Components: {rewards}")

if __name__ == "__main__":
    test_reward_functions()
