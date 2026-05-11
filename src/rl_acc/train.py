import os
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from acc_env import ACCEnv
from config import (
    TRAINING_TIMESTEPS, SAVE_FREQUENCY, LOG_FREQUENCY,
    MODEL_DIR, LOG_DIR, BATCH_SIZE, GAMMA, LEARNING_RATE,
    N_EPOCHS, CLIP_RANGE
)


def train_ppo_model():
    print("=" * 50)
    print("Training ACC Agent with PPO")
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = ACCEnv()
    eval_env = ACCEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=2048,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=CLIP_RANGE,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        policy_kwargs={
            'net_arch': [64, 64],
            'activation_fn': torch.nn.ReLU
        }
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQUENCY,
        save_path=MODEL_DIR,
        name_prefix="acc_model"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=SAVE_FREQUENCY // 2,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            log_interval=LOG_FREQUENCY // 10,
            tb_log_name="ppo_acc_training"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    final_model_path = os.path.join(MODEL_DIR, "final_model.zip")
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    env.close()
    eval_env.close()

    return model


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    model = train_ppo_model()

    print("\nTraining completed!")