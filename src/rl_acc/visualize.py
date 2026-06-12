import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from acc_env import ACCEnv
from config import MODEL_DIR


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


def collect_episode_data(model, env):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    return env.history.copy()


def plot_comprehensive_analysis(history, env, save_path='visualization_results'):
    os.makedirs(save_path, exist_ok=True)
    time_steps = np.arange(len(history['ego_speed'])) * 0.1

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time_steps, history['ego_speed'], 'b-', linewidth=2, label='Ego Speed')
    ax1.plot(time_steps, history['lead_speed'], 'g-', linewidth=2, label='Lead Speed')
    ax1.axhline(y=env.target_speed, color='r', linestyle='--', label='Target Speed')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Speed Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(time_steps, history['distance'], 'b-', linewidth=2, label='Actual Distance')
    desired_dist = env.safety_distance + np.array(history['ego_speed']) * 1.5
    ax2.plot(time_steps, desired_dist, 'r--', linewidth=2, label='Desired Distance')
    ax2.axhline(y=env.safety_distance, color='orange', linestyle='-.', label='Safety Distance')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Following Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(time_steps, history['acceleration'], 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.fill_between(time_steps, history['acceleration'], alpha=0.2, color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Acceleration Control')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(time_steps, history['reward'], 'm-', linewidth=1)
    cumulative_reward = np.cumsum(history['reward'])
    ax4.plot(time_steps, cumulative_reward, 'c-', linewidth=2, label='Cumulative')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Reward')
    ax4.set_title('Reward Signal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 2, 5)
    speed_error = np.abs(np.array(history['ego_speed']) - env.target_speed)
    ax5.plot(time_steps, speed_error, 'r-', linewidth=1)
    ax5.fill_between(time_steps, speed_error, alpha=0.3, color='red')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed Error (m/s)')
    ax5.set_title('Speed Tracking Error')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 2, 6)
    relative_speed = np.array(history['lead_speed']) - np.array(history['ego_speed'])
    ax6.scatter(history['distance'], history['ego_speed'], c=relative_speed, cmap='coolwarm', alpha=0.6, s=10)
    ax6.colorbar = plt.colorbar(ax6.collections[0], ax=ax6)
    ax6.set_xlabel('Distance (m)')
    ax6.set_ylabel('Ego Speed (m/s)')
    ax6.set_title('Phase Diagram (Speed vs Distance)')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comprehensive_analysis.png'), dpi=150)
    plt.close()

    print(f"Visualization saved to '{save_path}/'")


def main():
    model = load_model()
    env = ACCEnv()

    print("Collecting episode data...")
    history = collect_episode_data(model, env)

    print("Generating visualizations...")
    plot_comprehensive_analysis(history, env)

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()