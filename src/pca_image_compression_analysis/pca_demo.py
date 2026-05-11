"""PCA image compression on synthetic pattern images."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PCAResult:
    components: int
    reconstructed: np.ndarray
    mse: float
    explained_variance: float
    compression_ratio: float


def make_images(seed: int = 307, count: int = 260, size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images, labels = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for label in range(4):
        for _ in range(count // 4):
            if label == 0:
                image = np.exp(-((xx - rng.uniform(4, 12)) ** 2 + (yy - rng.uniform(4, 12)) ** 2) / rng.uniform(12, 25))
            elif label == 1:
                image = ((np.abs(xx - yy + rng.integers(-3, 4)) < 2)).astype(float)
            elif label == 2:
                image = ((np.sin((xx + rng.uniform(0, 4)) / rng.uniform(1.5, 3.0)) > 0)).astype(float)
            else:
                image = (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < rng.uniform(18, 38)).astype(float)
            image = np.clip(image + rng.normal(0, 0.12, image.shape), 0, 1)
            images.append(image.reshape(-1))
            labels.append(label)
    return np.array(images), np.array(labels)


def fit_pca(x: np.ndarray, components: int) -> PCAResult:
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:components]
    latent = centered @ basis.T
    reconstructed = np.clip(latent @ basis + mean, 0, 1)
    mse = float(np.mean((x - reconstructed) ** 2))
    explained = float(np.sum(singular[:components] ** 2) / np.sum(singular ** 2))
    ratio = float((components * (x.shape[1] + x.shape[0]) + x.shape[1]) / (x.shape[0] * x.shape[1]))
    return PCAResult(components, reconstructed, mse, explained, ratio)


def plot_reconstruction(x: np.ndarray, results: list[PCAResult], out: Path) -> Path:
    path = out / "pca_reconstruction_grid.png"
    fig, axes = plt.subplots(len(results) + 1, 8, figsize=(9, 1.5 * (len(results) + 1)))
    for i in range(8):
        axes[0, i].imshow(x[i].reshape(16, 16), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("clean")
    for row, result in enumerate(results, start=1):
        for i in range(8):
            axes[row, i].imshow(result.reconstructed[i].reshape(16, 16), cmap="gray", vmin=0, vmax=1)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(f"k={result.components}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_curves(results: list[PCAResult], out: Path) -> list[Path]:
    paths = []
    k = [r.components for r in results]
    plt.figure(figsize=(7.2, 5))
    plt.plot(k, [r.mse for r in results], marker="o", color="#2f80ed")
    plt.xlabel("PCA components")
    plt.ylabel("reconstruction MSE")
    plt.title("PCA reconstruction error")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    path = out / "pca_error_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    plt.figure(figsize=(7.2, 5))
    plt.plot(k, [r.explained_variance for r in results], marker="o", color="#27ae60", label="explained variance")
    plt.plot(k, [r.compression_ratio for r in results], marker="s", color="#eb5757", label="storage ratio")
    plt.xlabel("PCA components")
    plt.title("Variance retained and storage cost")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    path = out / "pca_variance_compression.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def run(output: Path, seed: int = 307) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    x, labels = make_images(seed=seed)
    results = [fit_pca(x, k) for k in [2, 4, 8, 16, 32]]
    files = [plot_reconstruction(x, [results[1], results[2], results[3]], output), *plot_curves(results, output)]
    csv_path = output / "pca_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("components,mse,explained_variance,compression_ratio\n")
        for r in results:
            f.write(f"{r.components},{r.mse:.6f},{r.explained_variance:.6f},{r.compression_ratio:.6f}\n")
    files.append(csv_path)
    best = results[3]
    metrics = {
        "project": "pca_image_compression_analysis",
        "sample_count": int(len(labels)),
        "image_size": 16,
        "best_components": best.components,
        "best_mse": round(best.mse, 6),
        "best_explained_variance": round(best.explained_variance, 4),
        "best_compression_ratio": round(best.compression_ratio, 4),
        "generated_files": [p.name for p in files],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pr_assets/pca_image_compression_analysis"),
        help="Directory for generated figures and metrics.",
    )
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
