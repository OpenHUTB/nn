import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from acc_env import ACCEnv
from config import TEST_EPISODES, MODEL_DIR


def load_model(model_path=None):
    if model_path is None:
        best_model_path = os.path.join(MODEL_DIR, "best_model.zip")
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            existing_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
            if existing_models:
                model_path = os.path.join(MODEL_DIR, max(existing_models, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x))))
            else:
                raise FileNotFoundError("No trained model found")

    print(f"Loading model: {model_path}")
    return PPO.load(model_path)


def test_model(model, env, num_episodes=5):
    results = {
        'total_rewards': [],
        'collisions': 0,
        'success_episodes': 0,
        'histories': []
    }

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)

        obs, _ = env.reset()
        episode_reward = 0
        done = False
        collision = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward

            if info.get('collision', False):
                collision = True

        results['histories'].append(env.history.copy())
        results['total_rewards'].append(episode_reward)

        if collision:
            results['collisions'] += 1
            print(f"Collision! Reward: {episode_reward:.2f}")
        else:
            results['success_episodes'] += 1
            print(f"Success! Reward: {episode_reward:.2f}")

    results['avg_reward'] = np.mean(results['total_rewards'])
    results['success_rate'] = results['success_episodes'] / num_episodes

    return results


def plot_results(results, save_path='test_results'):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.bar(range(len(results['total_rewards'])), results['total_rewards'])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.grid(True, alpha=0.3)

    success_idx = 0
    for i, history in enumerate(results['histories']):
        if len(history['ego_speed']) > 0:
            success_idx = i
            break

    history = results['histories'][success_idx]
    time_steps = np.arange(len(history['ego_speed'])) * 0.1

    plt.subplot(2, 2, 2)
    plt.plot(time_steps, history['ego_speed'], label='Ego Speed')
    plt.plot(time_steps, history['lead_speed'], label='Lead Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Tracking')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(time_steps, history['distance'], label='Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Following Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(time_steps, history['acceleration'], label='Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Acceleration Control')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_results.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to '{save_path}/' directory")


def main():
    parser = argparse.ArgumentParser(description='Test ACC Model')
    parser.add_argument('--model', type=str, default=None, help='Model path')
    parser.add_argument('--episodes', type=int, default=TEST_EPISODES, help='Number of episodes')
    args = parser.parse_args()

    model = load_model(args.model)
    env = ACCEnv()

    results = test_model(model, env, num_episodes=args.episodes)

    print("\n" + "=" * 40)
    print("Test Results Summary")
    print("=" * 40)
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Collisions: {results['collisions']}")

    plot_results(results)

    env.close()


if __name__ == "__main__":
    main()