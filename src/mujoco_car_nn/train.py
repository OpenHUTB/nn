"""Behaviour cloning / DAgger training loops."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CollectConfig, DrivingDataset, collect_dataset, split_indices
from .expert import PurePursuitExpert
from .models import build_model


@dataclass
class TrainConfig:
    model: str = "cnn"
    epochs: int = 24
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steer_weight: float = 1.0
    throttle_weight: float = 0.6
    augment: bool = True
    val_ratio: float = 0.15
    seed: int = 0
    device: str = "cpu"


def train_policy(
    data: dict[str, np.ndarray],
    config: TrainConfig | None = None,
    model: nn.Module | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    images, actions = data["images"], data["actions"].astype(np.float32)
    states = data.get("states")
    train_idx, val_idx = split_indices(len(images), cfg.val_ratio, cfg.seed)
    train_ds = DrivingDataset(
        images[train_idx], actions[train_idx], None if states is None else states[train_idx], augment=cfg.augment, seed=cfg.seed
    )
    val_ds = DrivingDataset(images[val_idx], actions[val_idx], None if states is None else states[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    net = model if model is not None else build_model(cfg.model)
    net.to(cfg.device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = torch.tensor([cfg.steer_weight, cfg.throttle_weight], device=cfg.device)

    history = {"train_loss": [], "val_loss": [], "val_steer_mae": [], "val_throttle_mae": []}
    best_state, best_val = None, float("inf")
    t0 = time.time()

    for epoch in range(cfg.epochs):
        net.train()
        total, count = 0.0, 0
        for x, s, y in train_loader:
            x, y, s = x.to(cfg.device), y.to(cfg.device), s.to(cfg.device)
            pred = net(x, s)
            loss = (((pred - y) ** 2) * weights).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(x)
            count += len(x)
        sched.step()

        net.eval()
        val_total, val_count = 0.0, 0
        abs_steer, abs_thr = 0.0, 0.0
        with torch.no_grad():
            for x, s, y in val_loader:
                x, y, s = x.to(cfg.device), y.to(cfg.device), s.to(cfg.device)
                pred = net(x, s)
                val_total += float((((pred - y) ** 2) * weights).mean()) * len(x)
                abs_steer += float((pred[:, 0] - y[:, 0]).abs().sum())
                abs_thr += float((pred[:, 1] - y[:, 1]).abs().sum())
                val_count += len(x)
        val_loss = val_total / max(val_count, 1)
        history["train_loss"].append(total / max(count, 1))
        history["val_loss"].append(val_loss)
        history["val_steer_mae"].append(abs_steer / max(val_count, 1))
        history["val_throttle_mae"].append(abs_thr / max(val_count, 1))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
        if verbose and (epoch % 4 == 0 or epoch == cfg.epochs - 1):
            print(
                f"epoch {epoch:3d}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}  "
                f"steer_mae {history['val_steer_mae'][-1]:.3f}  thr_mae {history['val_throttle_mae'][-1]:.3f}"
            )

    if best_state is not None:
        net.load_state_dict(best_state)
    history["train_seconds"] = time.time() - t0
    history["n_train"] = len(train_idx)
    history["n_val"] = len(val_idx)
    history["best_val"] = best_val
    return net, history


def dagger_train(
    track,
    data: dict[str, np.ndarray],
    config: TrainConfig | None = None,
    iterations: int = 2,
    collect: CollectConfig | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, np.ndarray], dict]:
    """DAgger: roll out the current policy, label the visited states with the
    expert, aggregate the data and retrain."""
    from .evaluate import rollout  # local import to avoid a cycle
    from .models import PolicyRunner

    cfg = config or TrainConfig()
    net, history = train_policy(data, cfg, verbose=verbose)
    logs = []

    for it in range(iterations):
        runner = PolicyRunner(net)
        new = rollout_policy_collect(track, runner, collect or CollectConfig(), seed=cfg.seed + 100 + it)
        data = {k: np.concatenate([data[k], new[k]], axis=0) for k in ("images", "actions", "states", "scans", "scenes")}
        net, hist = train_policy(data, cfg, verbose=verbose)
        logs.append(hist)
    return net, data, {"bc": history, "dagger": logs}


def rollout_policy_collect(track, policy, cfg: CollectConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Collect states visited by ``policy`` with expert labels (DAgger)."""
    from .sim import CarSim, SimConfig

    rng = np.random.default_rng(seed)
    sim = CarSim(track, SimConfig(scene=cfg.scenes[0], seed=seed, randomize_start=True))
    expert = PurePursuitExpert(track)
    st = sim.reset()
    images, actions, states, scans, scenes = [], [], [], [], []
    prev_s = st.s
    travelled = 0.0
    steps = 0
    while travelled < cfg.laps * track.length and steps < cfg.max_steps:
        img = sim.camera_image()
        label = expert.act(st)
        action = policy(img, st.speed)
        # beta-mixing keeps the car moving so that the visited state
        # distribution stays close to the expert's (classic DAgger trick).
        if rng.random() < 0.35:
            action = label.copy()
        if rng.random() < 0.15:
            action = action + rng.normal(0, 0.1, size=2)
        images.append(img)
        scans.append(sim.lidar())
        states.append(st.as_vector())
        actions.append(label)
        scenes.append(cfg.scenes[0])
        st = sim.step(np.clip(action, -1, 1))
        travelled += (st.s - prev_s) % track.length
        prev_s = st.s
        steps += 1
    sim.close()
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "scans": np.asarray(scans, dtype=np.float32),
        "scenes": np.asarray(scenes),
    }
