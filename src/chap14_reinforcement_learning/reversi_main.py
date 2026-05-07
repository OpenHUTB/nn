import json
import os
import random
from pathlib import Path

import numpy as np


def _env_int(name, default, minimum=1):
    """Read integer environment variables safely."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(value, minimum)


def _save_report(report_path, payload):
    """Save JSON report."""
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] report saved: {out_path.resolve()}")


# Configurable parameters
SEED = _env_int("REVERSI_SEED", 42, minimum=0)
MAX_EPOCHS = _env_int("REVERSI_MAX_EPOCHS", 100, minimum=1)
RENDER_INTERVAL = _env_int("REVERSI_RENDER_INTERVAL", 10, minimum=1)
MAX_STEPS_PER_EPISODE = _env_int("REVERSI_MAX_STEPS", 100, minimum=1)
REPORT_OUT = os.getenv("REVERSI_REPORT_OUT", "outputs/reversi_train_report.json")
DRY_RUN = os.getenv("REVERSI_DRY_RUN", "0").strip() == "1"

random.seed(SEED)
np.random.seed(SEED)


if DRY_RUN:
    _save_report(
        REPORT_OUT,
        {
            "dry_run": True,
            "config": {
                "seed": SEED,
                "max_epochs": MAX_EPOCHS,
                "render_interval": RENDER_INTERVAL,
                "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            },
            "note": "DRY_RUN enabled, training loop skipped.",
        },
    )
    raise SystemExit(0)


import gym
from gym.envs.registration import register

from RL_QG_agent import RL_QG_agent


ENV_ID = "Reversi8x8-v0"


def _ensure_registered():
    """Register env once."""
    env_ids = [spec.id for spec in gym.envs.registry.all()]
    if ENV_ID in env_ids:
        return
    register(
        id=ENV_ID,
        entry_point="gym.envs.reversi.reversi:ReversiEnv",
        kwargs={
            "player_color": "black",
            "opponent": "random",
            "observation_type": "numpy3c",
            "illegal_place_mode": "lose",
            "board_size": 8,
        },
        max_episode_steps=1000,
    )


def run_training():
    _ensure_registered()
    env = gym.make(
        ENV_ID,
        player_color="black",
        opponent="random",
        observation_type="numpy3c",
        illegal_place_mode="lose",
    )

    agent = RL_QG_agent()
    agent.init_model()
    agent.load_model()

    print(f"[INFO] {ENV_ID} registered: True")
    print(
        "[INFO] config:",
        {
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "render_interval": RENDER_INTERVAL,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "report_out": REPORT_OUT,
        },
    )

    episode_summaries = []
    for i_episode in range(MAX_EPOCHS):
        observation = env.reset()

        for t in range(MAX_STEPS_PER_EPISODE):
            if i_episode % RENDER_INTERVAL == 0:
                env.render()

            enables = env.possible_actions
            if len(enables) == 0:
                action_black = env.board_size**2 + 1
            else:
                action_black = random.choice(enables)

            observation, reward, done, info = env.step(action_black)
            if done:
                break

            if i_episode % RENDER_INTERVAL == 0:
                env.render()

            enables = env.possible_actions
            if not enables:
                action_white = env.board_size**2 + 1
            else:
                action_white = agent.place(observation, enables)

            observation, reward, done, info = env.step(action_white)
            if done:
                break

        black_score = int(np.sum(env.board == 1))
        white_score = int(np.sum(env.board == -1))

        if black_score > white_score:
            winner = "black"
        elif black_score < white_score:
            winner = "white"
        else:
            winner = "draw"

        print(
            f"Episode {i_episode + 1}/{MAX_EPOCHS}, steps={t + 1}, "
            f"black={black_score}, white={white_score}, winner={winner}"
        )

        episode_summaries.append(
            {
                "episode": i_episode + 1,
                "steps": t + 1,
                "black_score": black_score,
                "white_score": white_score,
                "winner": winner,
            }
        )

    agent.save_model()
    env.close()

    _save_report(
        REPORT_OUT,
        {
            "dry_run": False,
            "config": {
                "seed": SEED,
                "max_epochs": MAX_EPOCHS,
                "render_interval": RENDER_INTERVAL,
                "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            },
            "episodes": episode_summaries,
            "final_episode": episode_summaries[-1] if episode_summaries else None,
        },
    )


if __name__ == "__main__":
    run_training()
