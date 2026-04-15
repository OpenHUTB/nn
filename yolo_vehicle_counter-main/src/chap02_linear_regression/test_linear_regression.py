#!/usr/bin/env python
# coding: utf-8
"""
单元测试模块：线性回归模块的功能测试

本模块为 exercise-linear_regression.py 中的各个函数提供全面的单元测试，
包括数据加载、基函数变换、模型优化和评估等功能。

运行方式：
    pytest test_linear_regression.py -v
    或
    python test_linear_regression.py
"""

import sys
import os
from pathlib import Path

# 设置路径以便导入
sys.path.insert(0, str(Path(__file__).parent))

# 延迟导入，只在实际执行测试时导入
try:
    import numpy as np
    import pytest
    HAS_NUMPY = True
    HAS_PYTEST = True
except ImportError:
    HAS_NUMPY = False
    HAS_PYTEST = False

# 动态导入被测试的模块（处理包含破折号的文件名）
if HAS_NUMPY:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exercise_linear_regression",
        Path(__file__).parent / "exercise-linear_regression.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    # 临时禁用 matplotlib 显示
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        spec.loader.exec_module(module)
        
        # 从模块中导入所需函数
        load_data = module.load_data
        identity_basis = module.identity_basis
        multinomial_basis = module.multinomial_basis
        gaussian_basis = module.gaussian_basis
        least_squares = module.least_squares
        gradient_descent = module.gradient_descent
        main = module.main
        evaluate = module.evaluate
        MODULE_LOADED = True
    except Exception as e:
        print(f"警告：无法加载模块: {e}")
        MODULE_LOADED = False
else:
    MODULE_LOADED = False


# ============================================================================
# 单元测试类（仅当依赖模块可用时执行）
# ============================================================================

if HAS_PYTEST and MODULE_LOADED:
    class TestDataLoading:
    """测试数据加载功能"""

    def test_load_data_basic(self):
        """测试基础数据加载功能"""
        # 使用项目中的实际数据文件
        data_dir = Path(__file__).parent
        train_file = data_dir / "train.txt"
        
        if train_file.exists():
            xs, ys = load_data(str(train_file))
            # 验证返回值是 numpy 数组
            assert isinstance(xs, np.ndarray), "特征应为 numpy 数组"
            assert isinstance(ys, np.ndarray), "标签应为 numpy 数组"
            # 验证形状一致
            assert xs.shape[0] == ys.shape[0], "特征和标签的样本数应相同"
            # 验证数据非空
            assert xs.shape[0] > 0, "加载的数据不应为空"

    def test_load_data_shape(self):
        """测试加载数据的形状"""
        data_dir = Path(__file__).parent
        train_file = data_dir / "train.txt"
        
        if train_file.exists():
            xs, ys = load_data(str(train_file))
            # train.txt 应有 10 行数据
            assert xs.shape[0] == 10, "train.txt 应包含 10 个样本"
            assert ys.shape[0] == 10, "标签数应为 10"

    def test_load_data_values(self):
        """测试加载数据的数值范围"""
        data_dir = Path(__file__).parent
        train_file = data_dir / "train.txt"
        
        if train_file.exists():
            xs, ys = load_data(str(train_file))
            # 根据文档，特征范围应在 0-25 之间
            assert np.all(xs >= 0) and np.all(xs <= 30), "特征值应在合理范围内"
            # 标签应为正数
            assert np.all(ys > 0), "标签值应为正数"


class TestBasisFunctions:
    """测试各种基函数"""

    def test_identity_basis_shape(self):
        """测试恒等基函数的输出形状"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        phi = identity_basis(x)
        # 输出形状应为 (N, 1)
        assert phi.shape == (5, 1), f"恒等基函数输出形状应为 (5, 1)，实际为 {phi.shape}"

    def test_identity_basis_values(self):
        """测试恒等基函数的输出值"""
        x = np.array([1.0, 2.0, 3.0])
        phi = identity_basis(x)
        expected = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_array_almost_equal(phi, expected)

    def test_multinomial_basis_shape(self):
        """测试多项式基函数的输出形状"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feature_num = 10
        phi = multinomial_basis(x, feature_num=feature_num)
        # 输出形状应为 (N, feature_num)
        assert phi.shape == (5, feature_num), \
            f"多项式基函数输出形状应为 (5, {feature_num})，实际为 {phi.shape}"

    def test_multinomial_basis_values(self):
        """测试多项式基函数的具体数值"""
        x = np.array([2.0])
        phi = multinomial_basis(x, feature_num=3)
        # 应该得到 [2^1, 2^2, 2^3] = [2, 4, 8]
        expected = np.array([[2.0, 4.0, 8.0]])
        np.testing.assert_array_almost_equal(phi, expected)

    def test_multinomial_basis_default_parameter(self):
        """测试多项式基函数的默认参数"""
        x = np.array([1.0, 2.0, 3.0])
        phi = multinomial_basis(x)  # 使用默认参数 feature_num=10
        assert phi.shape == (3, 10), "默认参数应该生成 10 个特征"

    def test_gaussian_basis_shape(self):
        """测试高斯基函数的输出形状"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feature_num = 10
        phi = gaussian_basis(x, feature_num=feature_num)
        # 输出形状应为 (N, feature_num)
        assert phi.shape == (5, feature_num), \
            f"高斯基函数输出形状应为 (5, {feature_num})，实际为 {phi.shape}"

    def test_gaussian_basis_values_range(self):
        """测试高斯基函数的输出值范围"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        phi = gaussian_basis(x, feature_num=10)
        # 高斯函数的值应在 [0, 1] 范围内
        assert np.all(phi >= 0) and np.all(phi <= 1), \
            "高斯基函数的输出值应在 [0, 1] 范围内"

    def test_gaussian_basis_max_near_center(self):
        """测试高斯基函数的中心性"""
        x = np.array([12.5])  # 在中心区域
        phi = gaussian_basis(x, feature_num=10)
        # 至少有一个高斯分量的值应接近 1
        assert np.max(phi) > 0.5, "中心区域的高斯响应应该较强"


class TestOptimization:
    """测试优化算法"""

    def test_least_squares_basic(self):
        """测试最小二乘法的基本功能"""
        # 构造一个简单的线性问题：y = 2*x + 3
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3
        
        # 构造设计矩阵（第一列为偏置项 1，第二列为特征 x）
        phi = np.column_stack([np.ones_like(x), x])
        
        # 最小二乘法求解
        w = least_squares(phi, y)
        
        # 验证求解的权重接近 [3, 2]
        np.testing.assert_array_almost_equal(w, [3, 2], decimal=5)

    def test_least_squares_different_solvers(self):
        """测试不同求解器的一致性"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3
        phi = np.column_stack([np.ones_like(x), x])
        
        # 使用不同求解器求解
        w_pinv = least_squares(phi, y, solver="pinv")
        w_svd = least_squares(phi, y, solver="svd")
        
        # 不同求解器的结果应该接近
        np.testing.assert_array_almost_equal(w_pinv, w_svd, decimal=4)

    def test_least_squares_regularization(self):
        """测试正则化参数的效果"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3 + np.random.randn(5) * 0.1  # 添加噪声
        phi = np.column_stack([np.ones_like(x), x])
        
        # 使用不同的正则化参数求解
        w_no_reg = least_squares(phi, y, alpha=0.0)
        w_with_reg = least_squares(phi, y, alpha=0.1)
        
        # 两种情况都应该返回权重向量
        assert w_no_reg.shape == phi.shape[1:], "权重形状应正确"
        assert w_with_reg.shape == phi.shape[1:], "权重形状应正确"

    def test_least_squares_error_handling(self):
        """测试最小二乘法的错误处理"""
        # 测试维度不匹配的错误
        phi = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])  # 维度不匹配
        
        with pytest.raises(ValueError):
            least_squares(phi, y)

    def test_gradient_descent_convergence(self):
        """测试梯度下降是否收敛"""
        # 构造一个简单的线性问题
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3
        phi = np.column_stack([np.ones_like(x), x])
        
        # 梯度下降求解
        w = gradient_descent(phi, y, lr=0.01, epochs=1000)
        
        # 验证求解的权重接近 [3, 2]（允许较大的容差，因为梯度下降可能不完全收敛）
        np.testing.assert_array_almost_equal(w, [3, 2], decimal=1)

    def test_gradient_descent_different_lr(self):
        """测试不同学习率的影响"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 3
        phi = np.column_stack([np.ones_like(x), x])
        
        # 使用不同学习率
        w_lr1 = gradient_descent(phi, y, lr=0.001, epochs=10000)
        w_lr2 = gradient_descent(phi, y, lr=0.01, epochs=1000)
        
        # 都应该返回权重向量
        assert w_lr1.shape == (2,), "权重形状应为 (2,)"
        assert w_lr2.shape == (2,), "权重形状应为 (2,)"


class TestModelTraining:
    """测试模型训练和评估"""

    def test_main_basic_training(self):
        """测试模型基础训练功能"""
        # 创建简单的训练数据
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = 2 * x_train + 3
        
        # 训练模型
        f, w_lsq, w_gd = main(x_train, y_train)
        
        # 验证返回值
        assert callable(f), "返回的应是可调用的函数"
        assert w_lsq is not None, "应返回最小二乘权重"
        assert w_gd is None, "未使用梯度下降时，w_gd 应为 None"

    def test_main_with_basis_function(self):
        """测试使用不同基函数的模型训练"""
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = 2 * x_train + 3
        
        # 使用多项式基函数
        f, w_lsq, w_gd = main(x_train, y_train, basis_func=multinomial_basis)
        
        # 验证权重维度
        # 1（偏置项） + 10（多项式特征）
        assert w_lsq.shape[0] == 11, "多项式基函数应生成 11 个权重"

    def test_main_with_gradient_descent(self):
        """测试使用梯度下降的模型训练"""
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = 2 * x_train + 3
        
        # 训练模型，使用梯度下降
        f, w_lsq, w_gd = main(x_train, y_train, use_gradient_descent=True)
        
        # 验证返回值
        assert w_lsq is not None, "应返回最小二乘权重"
        assert w_gd is not None, "使用梯度下降时，w_gd 应不为 None"

    def test_main_prediction(self):
        """测试模型的预测功能"""
        x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = 2 * x_train + 3
        
        # 训练模型
        f, w_lsq, w_gd = main(x_train, y_train)
        
        # 进行预测
        x_test = np.array([1.5, 2.5, 3.5])
        y_pred = f(x_test)
        
        # 验证预测输出
        assert y_pred.shape == x_test.shape, "预测输出形状应与输入形状相同"
        assert np.all(np.isfinite(y_pred)), "预测值应为有限数"

    def test_evaluate_basic(self):
        """测试模型评估功能"""
        ys_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys_pred = np.array([1.1, 2.0, 2.9, 4.1, 4.9])
        
        # 评估模型
        error = evaluate(ys_true, ys_pred)
        
        # 验证错误是正数
        assert isinstance(error, (float, np.floating)), "错误应为浮点数"
        assert error > 0, "错误应为正数"

    def test_evaluate_perfect_prediction(self):
        """测试完美预测的评估"""
        ys_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 完美预测的错误应接近 0
        error = evaluate(ys_true, ys_pred)
        assert error < 1e-10, "完美预测的错误应接近 0"


class TestIntegration:
    """集成测试：完整的工作流程"""

    def test_full_workflow_with_data_files(self):
        """测试完整的工作流程（使用项目中的数据文件）"""
        data_dir = Path(__file__).parent
        train_file = data_dir / "train.txt"
        test_file = data_dir / "test.txt"
        
        if train_file.exists() and test_file.exists():
            # 加载数据
            x_train, y_train = load_data(str(train_file))
            x_test, y_test = load_data(str(test_file))
            
            # 训练模型（使用恒等基函数）
            f, w_lsq, _ = main(x_train, y_train)
            
            # 进行预测
            y_pred = f(x_test)
            
            # 评估模型
            error = evaluate(y_test, y_pred)
            
            # 验证结果
            assert y_pred.shape == y_test.shape, "预测和测试集大小应相同"
            assert error > 0, "错误应为正数"
            assert error < 100, "错误应在可接受范围内"

    def test_workflow_with_polynomial_basis(self):
        """测试使用多项式基函数的完整流程"""
        # 创建测试数据
        x_train = np.linspace(0, 25, 20)
        # 生成复杂的非线性关系
        y_train = x_train**2 / 100 + 2 * x_train + 5 + np.random.randn(20) * 0.5
        
        # 训练模型（使用多项式基函数）
        f, w_lsq, _ = main(x_train, y_train, basis_func=multinomial_basis)
        
        # 验证权重维度正确
        assert w_lsq.shape[0] == 11, "多项式基函数应生成 11 个权重"
        
        # 进行预测
        x_test = np.linspace(0, 25, 10)
        y_pred = f(x_test)
        
        # 评估模型
        y_train_pred = f(x_train)
        train_error = evaluate(y_train, y_train_pred)
        
        # 多项式基函数应能更好地拟合
        assert train_error < 2.0, "使用多项式基函数训练误差应较小"


if __name__ == "__main__":
    # 可以直接运行此文件进行测试
    pytest.main([__file__, "-v"])
