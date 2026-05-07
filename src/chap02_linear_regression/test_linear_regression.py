#!/usr/bin/env python
# coding: utf-8
"""Tests for the linear regression exercise module."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parent / "exercise-linear_regression.py"
spec = importlib.util.spec_from_file_location("exercise_linear_regression", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

load_data = module.load_data
identity_basis = module.identity_basis
multinomial_basis = module.multinomial_basis
gaussian_basis = module.gaussian_basis
least_squares = module.least_squares
gradient_descent = module.gradient_descent
build_design_matrix = module.build_design_matrix
main = module.main
evaluate = module.evaluate


class TestDataLoading:
    """Test data loading behavior."""

    def test_load_data_basic(self):
        data_file = Path(__file__).parent / "train.txt"
        xs, ys = load_data(data_file)

        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)
        assert xs.shape == ys.shape
        assert xs.size > 0

    def test_load_data_rejects_bad_column_count(self, tmp_path):
        data_file = tmp_path / "bad.txt"
        data_file.write_text("1.0 2.0 3.0\n", encoding="utf-8")

        with pytest.raises(ValueError, match="应包含 2 个数值"):
            load_data(data_file)

    def test_load_data_rejects_empty_file(self, tmp_path):
        data_file = tmp_path / "empty.txt"
        data_file.write_text("\n", encoding="utf-8")

        with pytest.raises(ValueError, match="没有可用的数据"):
            load_data(data_file)


class TestBasisFunctions:
    """Test basis function transforms."""

    def test_identity_basis_shape_and_values(self):
        x = np.array([1.0, 2.0, 3.0])
        phi = identity_basis(x)

        assert phi.shape == (3, 1)
        np.testing.assert_array_almost_equal(phi, np.array([[1.0], [2.0], [3.0]]))

    def test_multinomial_basis_shape_and_values(self):
        x = np.array([2.0])
        phi = multinomial_basis(x, feature_num=3)

        assert phi.shape == (1, 3)
        np.testing.assert_array_almost_equal(phi, np.array([[2.0, 4.0, 8.0]]))

    def test_multinomial_basis_rejects_invalid_feature_count(self):
        with pytest.raises(ValueError, match="feature_num"):
            multinomial_basis(np.array([1.0]), feature_num=0)

    def test_gaussian_basis_shape_and_range(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        phi = gaussian_basis(x, feature_num=10)

        assert phi.shape == (5, 10)
        assert np.all(phi >= 0)
        assert np.all(phi <= 1)

    def test_gaussian_basis_rejects_invalid_range(self):
        with pytest.raises(ValueError, match="max_value"):
            gaussian_basis(np.array([1.0]), min_value=1.0, max_value=1.0)


class TestOptimization:
    """Test optimization algorithms."""

    def test_least_squares_solves_simple_line(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3
        phi = np.column_stack([np.ones_like(x), x])

        w = least_squares(phi, y)

        np.testing.assert_array_almost_equal(w, np.array([3.0, 2.0]), decimal=5)

    def test_least_squares_solvers_are_consistent(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3
        phi = np.column_stack([np.ones_like(x), x])

        w_pinv = least_squares(phi, y, solver="pinv")
        w_svd = least_squares(phi, y, solver="svd")
        w_cholesky = least_squares(phi, y, alpha=0.01, solver="cholesky")

        np.testing.assert_array_almost_equal(w_pinv, w_svd, decimal=4)
        assert w_cholesky.shape == (2,)

    def test_least_squares_rejects_bad_shapes(self):
        phi = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="不匹配"):
            least_squares(phi, y)

    def test_gradient_descent_converges_on_simple_line(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3
        phi = np.column_stack([np.ones_like(x), x])

        w = gradient_descent(phi, y, lr=0.01, epochs=1000)

        np.testing.assert_array_almost_equal(w, np.array([3.0, 2.0]), decimal=1)

    def test_gradient_descent_rejects_invalid_learning_rate(self):
        phi = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="学习率"):
            gradient_descent(phi, y, lr=0.0)


class TestModelTraining:
    """Test model training and prediction."""

    def test_build_design_matrix_adds_bias(self):
        x = np.array([1.0, 2.0, 3.0])
        phi = build_design_matrix(x, identity_basis)

        assert phi.shape == (3, 2)
        np.testing.assert_array_equal(phi[:, 0], np.ones(3))

    def test_main_returns_prediction_function(self):
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = 2 * x_train + 3

        predict, w_lsq, w_gd = main(x_train, y_train)
        y_pred = predict(np.array([1.5, 2.5, 3.5]))

        assert callable(predict)
        assert w_lsq is not None
        assert w_gd is None
        assert y_pred.shape == (3,)
        assert np.all(np.isfinite(y_pred))

    def test_main_with_gradient_descent_returns_gd_weights(self):
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = 2 * x_train + 3

        _, w_lsq, w_gd = main(x_train, y_train, use_gradient_descent=True)

        assert w_lsq is not None
        assert w_gd is not None

    def test_evaluate_returns_rmse(self):
        ys_true = np.array([1.0, 2.0, 3.0])
        ys_pred = np.array([1.0, 2.0, 5.0])

        error = evaluate(ys_true, ys_pred)

        assert error == pytest.approx(np.sqrt(4.0 / 3.0))

    def test_evaluate_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="形状"):
            evaluate(np.array([1.0]), np.array([1.0, 2.0]))


class TestIntegration:
    """Test complete workflows."""

    def test_full_workflow_with_data_files(self):
        data_dir = Path(__file__).parent
        x_train, y_train = load_data(data_dir / "train.txt")
        x_test, y_test = load_data(data_dir / "test.txt")

        predict, _, _ = main(x_train, y_train)
        y_pred = predict(x_test)
        error = evaluate(y_test, y_pred)

        assert y_pred.shape == y_test.shape
        assert error > 0
        assert error < 100

    def test_workflow_with_polynomial_basis(self):
        rng = np.random.default_rng(0)
        x_train = np.linspace(0, 25, 20)
        y_train = x_train**2 / 100 + 2 * x_train + 5 + rng.normal(0, 0.2, 20)

        predict, w_lsq, _ = main(x_train, y_train, basis_func=multinomial_basis)
        train_error = evaluate(y_train, predict(x_train))

        assert w_lsq.shape[0] == 11
        assert train_error < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
