#!/usr/bin/env python
# coding: utf-8
"""
线性回归模块功能验证脚本

这是一个独立的测试脚本，用于验证 exercise-linear_regression.py 的核心功能。
无需额外的测试框架依赖，可以直接运行。

运行方式：
    python test_verify.py
"""

import sys
import os
import argparse
import json
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent))

# 尝试导入必要的包
try:
    import numpy as np
except ImportError:
    print("错误: numpy 未安装。请运行: pip install numpy")
    sys.exit(1)

# 不导入 matplotlib，只在需要时处理
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exercise_linear_regression",
    Path(__file__).parent / "exercise-linear_regression.py"
)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
except Exception as e:
    print(f"错误: 无法加载模块 - {e}")
    sys.exit(1)

# 导出函数
load_data = module.load_data
identity_basis = module.identity_basis
multinomial_basis = module.multinomial_basis
gaussian_basis = module.gaussian_basis
least_squares = module.least_squares
gradient_descent = module.gradient_descent
main = module.main
evaluate = module.evaluate


def test_identity_basis():
    """测试恒等基函数"""
    print("测试: identity_basis...", end=" ")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    phi = identity_basis(x)
    assert phi.shape == (5, 1), f"Expected (5, 1), got {phi.shape}"
    assert np.allclose(phi.squeeze(), x), "输出值不匹配"
    print("[PASS] 通过")


def test_multinomial_basis():
    """测试多项式基函数"""
    print("测试: multinomial_basis...", end=" ")
    x = np.array([2.0])
    phi = multinomial_basis(x, feature_num=3)
    expected = np.array([[2.0, 4.0, 8.0]])
    assert phi.shape == (1, 3), f"Expected (1, 3), got {phi.shape}"
    assert np.allclose(phi, expected), "输出值不匹配"
    print("[PASS] 通过")


def test_gaussian_basis():
    """测试高斯基函数"""
    print("测试: gaussian_basis...", end=" ")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    phi = gaussian_basis(x, feature_num=10)
    assert phi.shape == (5, 10), f"Expected (5, 10), got {phi.shape}"
    assert np.all(phi >= 0) and np.all(phi <= 1), "高斯函数值应在 [0, 1] 范围内"
    print("[PASS] 通过")


def test_least_squares():
    """测试最小二乘法"""
    print("测试: least_squares...", end=" ")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 3
    phi = np.column_stack([np.ones_like(x), x])
    w = least_squares(phi, y)
    expected_w = np.array([3, 2])
    assert w.shape == (2,), f"Expected (2,), got {w.shape}"
    assert np.allclose(w, expected_w, atol=1e-5), f"Expected {expected_w}, got {w}"
    print("[PASS] 通过")


def test_gradient_descent():
    """测试梯度下降"""
    print("测试: gradient_descent...", end=" ")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 3
    phi = np.column_stack([np.ones_like(x), x])
    w = gradient_descent(phi, y, lr=0.01, epochs=1000)
    assert w.shape == (2,), f"Expected (2,), got {w.shape}"
    # 检查收敛性（允许较大容差）
    assert np.abs(w[0] - 3) < 1.0 and np.abs(w[1] - 2) < 1.0, "梯度下降未收敛"
    print("[PASS] 通过")


def test_main_function():
    """测试主训练函数"""
    print("测试: main function...", end=" ")
    x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_train = 2 * x_train + 3
    f, w_lsq, w_gd = main(x_train, y_train)
    assert callable(f), "返回值应该是可调用的函数"
    assert w_lsq is not None, "应返回最小二乘权重"
    assert w_gd is None, "未使用梯度下降时应返回 None"
    print("[PASS] 通过")


def test_main_with_basis():
    """测试使用不同基函数的主函数"""
    print("测试: main with multinomial_basis...", end=" ")
    x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_train = 2 * x_train + 3
    f, w_lsq, w_gd = main(x_train, y_train, basis_func=multinomial_basis)
    assert w_lsq.shape[0] == 11, "多项式基函数应生成 11 个权重（1个偏置 + 10个特征）"
    print("[PASS] 通过")


def test_main_prediction():
    """测试模型预测"""
    print("测试: main prediction...", end=" ")
    x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_train = 2 * x_train + 3
    f, w_lsq, w_gd = main(x_train, y_train)
    x_test = np.array([1.5, 2.5, 3.5])
    y_pred = f(x_test)
    assert y_pred.shape == x_test.shape, "预测输出形状应与输入相同"
    assert np.all(np.isfinite(y_pred)), "预测值应为有限数"
    print("[PASS] 通过")


def test_evaluate():
    """测试评估函数"""
    print("测试: evaluate...", end=" ")
    ys_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ys_pred = np.array([1.1, 2.0, 2.9, 4.1, 4.9])
    error = evaluate(ys_true, ys_pred)
    assert isinstance(error, (float, np.floating)), "错误应为浮点数"
    assert error > 0, "错误应为正数"
    print("[PASS] 通过")


def test_evaluate_perfect():
    """测试完美预测评估"""
    print("测试: evaluate (perfect prediction)...", end=" ")
    ys_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ys_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    error = evaluate(ys_true, ys_pred)
    assert error < 1e-10, "完美预测的错误应接近 0"
    print("[PASS] 通过")


def test_load_data():
    """测试数据加载"""
    print("测试: load_data...", end=" ")
    data_file = Path(__file__).parent / "train.txt"
    if data_file.exists():
        xs, ys = load_data(str(data_file))
        assert isinstance(xs, np.ndarray), "特征应为 numpy 数组"
        assert isinstance(ys, np.ndarray), "标签应为 numpy 数组"
        assert xs.shape[0] == ys.shape[0], "特征和标签数量应相同"
        print("[PASS] 通过")
    else:
        print("[SKIP] 跳过 (data file not found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression module verify script")
    parser.add_argument("--json-out", type=str, default=None, help="Optional: write test report to JSON file")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop immediately on first failure")
    args = parser.parse_args()

    print("=" * 60)
    print("Linear Regression Module Unit Tests")
    print("=" * 60)

    tests = [
        test_identity_basis,
        test_multinomial_basis,
        test_gaussian_basis,
        test_least_squares,
        test_gradient_descent,
        test_main_function,
        test_main_with_basis,
        test_main_prediction,
        test_evaluate,
        test_evaluate_perfect,
        test_load_data,
    ]

    passed = 0
    failed = 0
    results = []

    for test_func in tests:
        try:
            test_func()
            passed += 1
            results.append({"name": test_func.__name__, "status": "passed"})
        except Exception as e:
            print(f"[FAIL] Failed: {e}")
            failed += 1
            results.append({"name": test_func.__name__, "status": "failed", "error": str(e)})
            if args.stop_on_fail:
                break

    print("=" * 60)
    print(f"Test summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = Path(__file__).parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved: {out_path}")

    sys.exit(0 if failed == 0 else 1)
