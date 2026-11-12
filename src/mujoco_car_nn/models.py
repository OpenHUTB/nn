"""Neural network policies for end-to-end and modular driving.

The end-to-end policy consumes the 64x64 ego camera image and one
proprioceptive scalar (forward speed).  Speed is needed because a single
frame does not tell the network whether the car is cruising or standing
still, which makes the throttle label ambiguous (see ``docs/experiments``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PolicyCNN(nn.Module):
    """End-to-end CNN: 64x64 RGB image (+ speed) -> (steer, throttle)."""

    def __init__(self, in_channels: int = 3, out_dim: int = 2, state_dim: int = 1, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 + state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        h = self.features(x).flatten(1)
        if self.state_dim:
            h = torch.cat([h, state], dim=1)
        return self.head(h)


class PolicyMLP(nn.Module):
    """Same inputs, no spatial inductive bias (architecture ablation)."""

    def __init__(self, in_channels: int = 3, size: int = 64, out_dim: int = 2, state_dim: int = 1, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * size * size + state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        h = x.flatten(1)
        if self.state_dim:
            h = torch.cat([h, state], dim=1)
        return self.net(h)


class PerceptionCNN(nn.Module):
    """Image -> (lateral error, heading error), both normalised to [-1, 1]."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ModularPolicy(nn.Module):
    """Modular pipeline: perception CNN + planning/control MLP.

    The perception head predicts the lane-relative pose, the controller maps
    ``(lateral, heading, speed)`` to commands.  Both halves are trained
    separately (perception with ground-truth labels, controller with expert
    actions), which keeps every intermediate signal interpretable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 1
        self.perception = PerceptionCNN()
        self.controller = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        perception = self.perception(x)
        return self.controller(torch.cat([perception, state], dim=1))

    def errors(self, x: torch.Tensor) -> torch.Tensor:
        return self.perception(x)


def build_model(name: str, state_dim: int = 1) -> nn.Module:
    name = name.lower()
    if name in ("cnn", "e2e_cnn", "policy_cnn"):
        return PolicyCNN(state_dim=state_dim)
    if name in ("mlp", "e2e_mlp", "policy_mlp"):
        return PolicyMLP(state_dim=state_dim)
    if name in ("modular", "perception"):
        return ModularPolicy()
    raise ValueError(f"unknown model: {name}")


def preprocess(images: np.ndarray) -> torch.Tensor:
    """uint8 NHWC images -> float32 NCHW tensors in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(images)).float().div_(127.5).sub_(1.0)
    return x.permute(0, 3, 1, 2)


class PolicyRunner:
    """Convenience wrapper: numpy image + speed in, normalised actions out."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image: np.ndarray, speed: float = 0.0, max_speed: float = 16.0) -> np.ndarray:
        x = preprocess(image[None].astype(np.uint8)).to(self.device)
        state = None
        if getattr(self.model, "state_dim", 0):
            state = torch.tensor([[speed / max_speed]], dtype=torch.float32, device=self.device)
        return self.model(x, state).cpu().numpy()[0]


class ModularGeometricRunner:
    """Perception CNN + geometric pure-pursuit planner/controller.

    The network supplies the lane-relative pose, the planner turns it into a
    curvature command, and a P controller tracks the target speed.  This is
    the classic "learned perception + model-based control" stack.
    """

    def __init__(self, perception: nn.Module, wheelbase: float = 2.7, max_steer: float = 0.55):
        self.perception = perception.eval()
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    @torch.no_grad()
    def __call__(self, image: np.ndarray, speed: float = 0.0) -> np.ndarray:
        x = preprocess(image[None].astype(np.uint8))
        err = self.perception(x).numpy()[0]
        e_y, e_psi = float(err[0]) * 4.5, float(err[1]) * 0.6
        lookahead = float(np.clip(4.0 + 0.55 * speed, 3.0, 11.0))
        alpha = -(e_psi + float(np.arctan2(e_y, lookahead)))
        curvature = 2.0 * np.sin(alpha) / lookahead
        steer = float(np.clip(np.arctan(curvature * self.wheelbase) / self.max_steer, -1.0, 1.0))
        v_target = float(np.clip(9.0 - 6.0 * abs(e_psi), 4.0, 11.0))
        throttle = float(np.clip(0.3 + 0.5 * (v_target - speed), -1.0, 1.0))
        return np.array([steer, throttle], dtype=np.float32)
