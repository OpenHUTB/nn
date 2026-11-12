"""Dataset collection (expert driving) and torch datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from .expert import ExpertConfig, PurePursuitExpert
from .sim import IMAGE_SIZE, CarSim, SimConfig


@dataclass
class CollectConfig:
    laps: float = 6.0
    scenes: tuple[str, ...] = ("day",)
    seed: int = 0
    control_dt: float = 0.05
    perturb_prob: float = 0.12
    perturb_amp: float = 0.18
    randomize_start: bool = True
    offset_range: float = 1.2
    max_steps: int = 6000


def collect_dataset(track, config: CollectConfig | None = None, progress=None) -> dict[str, np.ndarray]:
    """Drive the expert around the track and record (image, lidar, state, action)."""
    cfg = config or CollectConfig()
    rng = np.random.default_rng(cfg.seed)
    images, actions, states, scans, scenes = [], [], [], [], []

    for scene in cfg.scenes:
        sim = CarSim(
            track,
            SimConfig(scene=scene, seed=cfg.seed, randomize_start=cfg.randomize_start, offset_range=cfg.offset_range),
        )
        expert = PurePursuitExpert(track)
        st = sim.reset()
        prev_s = st.s
        travelled = 0.0
        steps = 0
        while travelled < cfg.laps * track.length and steps < cfg.max_steps:
            action = expert.act(st)
            images.append(sim.camera_image())
            scans.append(sim.lidar())
            states.append(st.as_vector())
            actions.append(action)
            scenes.append(scene)

            executed = action.copy()
            if rng.random() < cfg.perturb_prob:
                executed[0] = float(np.clip(executed[0] + rng.normal(0, cfg.perturb_amp), -1, 1))
            if rng.random() < cfg.perturb_prob * 0.5:
                executed[1] = float(np.clip(executed[1] + rng.normal(0, 0.25), -1, 1))
            st = sim.step(executed)
            travelled += (st.s - prev_s) % track.length
            prev_s = st.s
            steps += 1
            if progress and steps % 200 == 0:
                progress(steps, travelled / track.length)
        sim.close()

    return {
        "images": np.asarray(images, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "scans": np.asarray(scans, dtype=np.float32),
        "scenes": np.asarray(scenes),
    }


def save_dataset(path: str, data: dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **data)


class DrivingDataset(Dataset):
    """Image + speed + action dataset with light photometric augmentation."""

    def __init__(
        self,
        images: np.ndarray,
        actions: np.ndarray,
        states: np.ndarray | None = None,
        augment: bool = False,
        seed: int = 0,
    ):
        self.images = images
        self.actions = actions.astype(np.float32)
        self.states = None if states is None else states.astype(np.float32)[:, :1]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx].astype(np.float32)
        if self.augment:
            img = img * self.rng.uniform(0.85, 1.15)
            img = np.clip(img + self.rng.uniform(-12, 12), 0, 255)
        x = torch.from_numpy(img).permute(2, 0, 1).div_(127.5).sub_(1.0)
        y = torch.from_numpy(self.actions[idx])
        if self.states is None:
            s = torch.zeros(1)
        else:
            s = torch.from_numpy(self.states[idx])
        return x, s, y


def load_dataset(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def split_indices(n: int, val_ratio: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]
