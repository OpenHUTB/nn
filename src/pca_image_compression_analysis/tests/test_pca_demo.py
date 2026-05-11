from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pca_demo import fit_pca, make_images, run


def test_more_components_reduce_error() -> None:
    x, _ = make_images(seed=307)
    small = fit_pca(x, 4)
    large = fit_pca(x, 16)
    assert large.mse < small.mse
    assert large.explained_variance > small.explained_variance


def test_exports(tmp_path: Path) -> None:
    metrics = run(tmp_path)
    assert metrics["best_explained_variance"] > 0.75
    assert (tmp_path / "pca_reconstruction_grid.png").exists()
    assert (tmp_path / "pca_error_curve.png").exists()
    assert (tmp_path / "pca_variance_compression.png").exists()


if __name__ == "__main__":
    test_more_components_reduce_error()
    with tempfile.TemporaryDirectory() as tmp:
        test_exports(Path(tmp))
    print("pca_image_compression_analysis tests passed")
