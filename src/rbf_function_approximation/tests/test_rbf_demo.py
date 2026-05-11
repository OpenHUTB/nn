from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rbf_demo import fit_linear_baseline, fit_rbf, make_dataset, mean_squared_error, predict_linear, predict_rbf, run


def test_rbf_beats_linear_baseline() -> None:
    data = make_dataset(seed=811)
    linear_w = fit_linear_baseline(data.x_train, data.y_train)
    rbf = fit_rbf(data.x_train, data.y_train)
    linear_mse = mean_squared_error(data.y_test, predict_linear(data.x_test, linear_w))
    rbf_mse = mean_squared_error(data.y_test, predict_rbf(data.x_test, rbf))
    assert rbf_mse < linear_mse * 0.35


def test_exports(tmp_path: Path) -> None:
    metrics = run(tmp_path)
    assert metrics["mse_reduction"] > 0.65
    assert (tmp_path / "rbf_fit_comparison.png").exists()
    assert (tmp_path / "rbf_basis_functions.png").exists()
    assert (tmp_path / "rbf_metric_comparison.png").exists()
    assert (tmp_path / "rbf_predictions.csv").exists()


if __name__ == "__main__":
    test_rbf_beats_linear_baseline()
    with tempfile.TemporaryDirectory() as tmp:
        test_exports(Path(tmp))
    print("rbf_function_approximation tests passed")
