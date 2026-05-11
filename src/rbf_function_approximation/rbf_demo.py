"""RBF network function approximation demo on synthetic regression data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_grid: np.ndarray
    y_true: np.ndarray


@dataclass
class RBFModel:
    centers: np.ndarray
    gamma: float
    weights: np.ndarray


def target_function(x: np.ndarray) -> np.ndarray:
    """Nonlinear target with smooth and local components."""
    return np.sin(2.2 * x) + 0.35 * np.cos(5.7 * x) + 0.18 * np.exp(-2.5 * (x - 1.1) ** 2)


def make_dataset(seed: int = 811, train_size: int = 90, test_size: int = 220) -> Dataset:
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(-3.0, 3.0, train_size))
    x_test = np.linspace(-3.0, 3.0, test_size)
    noise = rng.normal(0.0, 0.08, train_size)
    y_train = target_function(x_train) + noise
    y_test = target_function(x_test)
    x_grid = np.linspace(-3.2, 3.2, 420)
    return Dataset(x_train, y_train, x_test, y_test, x_grid, target_function(x_grid))


def design_matrix(x: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    dist2 = (x[:, None] - centers[None, :]) ** 2
    return np.exp(-gamma * dist2)


def fit_linear_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    features = np.column_stack([np.ones_like(x), x])
    return np.linalg.solve(features.T @ features + 1e-8 * np.eye(2), features.T @ y)


def predict_linear(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x]) @ weights


def fit_rbf(x: np.ndarray, y: np.ndarray, centers: int = 18, gamma: float = 2.6, ridge: float = 1e-3) -> RBFModel:
    center_points = np.linspace(float(x.min()), float(x.max()), centers)
    phi = design_matrix(x, center_points, gamma)
    features = np.column_stack([np.ones(len(x)), phi])
    weights = np.linalg.solve(features.T @ features + ridge * np.eye(features.shape[1]), features.T @ y)
    return RBFModel(center_points, gamma, weights)


def predict_rbf(x: np.ndarray, model: RBFModel) -> np.ndarray:
    phi = design_matrix(x, model.centers, model.gamma)
    return np.column_stack([np.ones(len(x)), phi]) @ model.weights


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def plot_fit(data: Dataset, linear_w: np.ndarray, rbf: RBFModel, output: Path) -> Path:
    path = output / "rbf_fit_comparison.png"
    plt.figure(figsize=(8.2, 5.2))
    plt.scatter(data.x_train, data.y_train, s=22, color="#4f4f4f", alpha=0.72, label="training samples")
    plt.plot(data.x_grid, data.y_true, color="#111827", linewidth=2.2, label="true function")
    plt.plot(data.x_grid, predict_linear(data.x_grid, linear_w), color="#eb5757", linewidth=2, label="linear baseline")
    plt.plot(data.x_grid, predict_rbf(data.x_grid, rbf), color="#2f80ed", linewidth=2.2, label="RBF network")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("RBF network nonlinear function approximation")
    plt.grid(True, linestyle="--", alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_basis(model: RBFModel, output: Path) -> Path:
    path = output / "rbf_basis_functions.png"
    x = np.linspace(-3.2, 3.2, 420)
    phi = design_matrix(x, model.centers, model.gamma)
    plt.figure(figsize=(8.2, 5.2))
    for i in range(phi.shape[1]):
        plt.plot(x, phi[:, i], color="#27ae60", alpha=0.28)
    plt.scatter(model.centers, np.ones_like(model.centers), color="#111827", s=16, zorder=3, label="centers")
    plt.xlabel("x")
    plt.ylabel("basis activation")
    plt.title("Radial basis activations and centers")
    plt.grid(True, linestyle="--", alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metrics(metrics: dict[str, float], output: Path) -> Path:
    path = output / "rbf_metric_comparison.png"
    labels = ["MSE", "MAE"]
    linear = [metrics["linear_mse"], metrics["linear_mae"]]
    rbf = [metrics["rbf_mse"], metrics["rbf_mae"]]
    x = np.arange(len(labels))
    width = 0.34
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - width / 2, linear, width, label="linear", color="#eb5757")
    plt.bar(x + width / 2, rbf, width, label="RBF", color="#2f80ed")
    plt.xticks(x, labels)
    plt.ylabel("error")
    plt.title("Generalization error comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def run(output: Path, seed: int = 811) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    data = make_dataset(seed=seed)
    linear_w = fit_linear_baseline(data.x_train, data.y_train)
    rbf = fit_rbf(data.x_train, data.y_train)
    linear_pred = predict_linear(data.x_test, linear_w)
    rbf_pred = predict_rbf(data.x_test, rbf)
    metrics = {
        "linear_mse": mean_squared_error(data.y_test, linear_pred),
        "linear_mae": mean_absolute_error(data.y_test, linear_pred),
        "rbf_mse": mean_squared_error(data.y_test, rbf_pred),
        "rbf_mae": mean_absolute_error(data.y_test, rbf_pred),
    }
    files = [
        plot_fit(data, linear_w, rbf, output),
        plot_basis(rbf, output),
        plot_metrics(metrics, output),
    ]
    csv_path = output / "rbf_predictions.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("x,true_y,linear_pred,rbf_pred\n")
        for x, y, lp, rp in zip(data.x_test, data.y_test, linear_pred, rbf_pred):
            f.write(f"{x:.6f},{y:.6f},{lp:.6f},{rp:.6f}\n")
    files.append(csv_path)
    report = {
        "project": "rbf_function_approximation",
        "train_size": int(len(data.x_train)),
        "test_size": int(len(data.x_test)),
        "rbf_centers": int(len(rbf.centers)),
        "gamma": rbf.gamma,
        "linear_mse": round(metrics["linear_mse"], 6),
        "rbf_mse": round(metrics["rbf_mse"], 6),
        "mse_reduction": round(1.0 - metrics["rbf_mse"] / metrics["linear_mse"], 4),
        "linear_mae": round(metrics["linear_mae"], 6),
        "rbf_mae": round(metrics["rbf_mae"], 6),
        "generated_files": [p.name for p in files],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pr_assets/rbf_function_approximation"),
        help="Directory for generated figures and metrics.",
    )
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
