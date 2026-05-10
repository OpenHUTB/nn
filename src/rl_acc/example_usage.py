import numpy as np
from stable_baselines3 import PPO

from acc_env import ACCEnv
from config import TARGET_SPEED, SAFETY_DISTANCE


def random_policy_demo():
    print("=" * 50)
    print("Random Policy Demo")
    print("=" * 50)

    env = ACCEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
    env.close()


def constant_speed_demo():
    print("\n" + "=" * 50)
    print("Constant Speed Policy Demo")
    print("=" * 50)

    env = ACCEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 500:
        ego_speed, lead_speed, distance, relative_speed, target_speed = obs

        if ego_speed < target_speed:
            acceleration = 1.0
        else:
            acceleration = -0.5

        desired_distance = SAFETY_DISTANCE + ego_speed * 1.5
        if distance < desired_distance:
            acceleration = min(acceleration, -1.0)

        action = np.array([acceleration])
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
    env.close()


def trained_model_demo():
    print("\n" + "=" * 50)
    print("Trained Model Demo")
    print("=" * 50)

    try:
        model = PPO.load("models/best_model.zip")
    except:
        print("No trained model found. Please train a model first: python train.py")
        return

    env = ACCEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
    env.close()


def main():
    print("RL-ACC Example Usage")
    print("=" * 50)

    random_policy_demo()
    constant_speed_demo()
    trained_model_demo()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)
    print("\nTo train your own model:")
    print("  python train.py")
    print("\nTo test the model:")
    print("  python test.py")
    print("\nTo visualize results:")
    print("  python visualize.py")


if __name__ == "__main__":
    main()