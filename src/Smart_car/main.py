#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的无人车速度预测系统
使用移动平均和线性回归方法进行速度预测
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings


def setup_matplotlib_for_plotting():
    """
    设置matplotlib和seaborn以确保正确的图表渲染
    在创建任何图表之前调用此函数
    """
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS",
                                       "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False


class SimpleSpeedPredictor:
    """简单的无人车速度预测器"""

    def __init__(self):
        self.method = "moving_average"
        self.window_size = 5

    def set_method(self, method, **kwargs):
        """设置预测方法"""
        self.method = method
        if method == "moving_average":
            self.window_size = kwargs.get('window_size', 5)
        elif method == "linear_regression":
            self.history_size = kwargs.get('history_size', 10)

    def moving_average_predict(self, speeds, window_size=5):
        """移动平均预测"""
        predictions = []
        for i in range(len(speeds)):
            if i < window_size:
                # 初期使用可用数据的平均值
                predictions.append(np.mean(speeds[:i + 1]))
            else:
                # 使用滑动窗口的平均值
                predictions.append(np.mean(speeds[i - window_size:i]))
        return np.array(predictions)

    def linear_regression_predict(self, speeds, history_size=10):
        """线性回归预测"""
        predictions = []

        for i in range(len(speeds)):
            if i < history_size:
                # 初期使用可用数据的平均值
                predictions.append(np.mean(speeds[:i + 1]))
            else:
                # 使用历史数据进行线性回归预测
                X = np.arange(i - history_size, i).reshape(-1, 1)
                y = speeds[i - history_size:i]

                # 拟合线性回归模型
                model = LinearRegression()
                model.fit(X, y)

                # 预测下一个值
                next_x = np.array([[i]])
                pred = model.predict(next_x)[0]
                predictions.append(pred)

        return np.array(predictions)

    def predict(self, speeds):
        """执行预测"""
        if self.method == "moving_average":
            return self.moving_average_predict(speeds, self.window_size)
        elif self.method == "linear_regression":
            return self.linear_regression_predict(speeds, self.history_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def generate_vehicle_data(num_points=50, base_speed=30, noise_level=2):
    """生成模拟的无人车速度数据"""
    np.random.seed(42)  # 确保结果可重现

    # 生成时间点
    time_points = np.arange(num_points)

    # 生成基础速度趋势（模拟加速、减速、匀速阶段）
    base_trend = np.zeros(num_points)

    # 0-10: 加速阶段
    base_trend[0:10] = base_speed + (time_points[0:10] * 2)
    # 10-30: 匀速阶段
    base_trend[10:30] = base_speed + 20 + np.random.normal(0, 1, 20)
    # 30-40: 减速阶段
    base_trend[30:40] = base_speed + 20 - (time_points[30:40] - 30) * 1.5
    # 40-50: 重新加速
    base_trend[40:50] = base_speed + 5 + (time_points[40:50] - 40) * 0.8

    # 添加噪声
    noise = np.random.normal(0, noise_level, num_points)
    speeds = base_trend + noise

    # 确保速度为正数且在合理范围内
    speeds = np.clip(speeds, 5, 80)

    return time_points, speeds


def evaluate_prediction(actual_speeds, predicted_speeds):
    """评估预测性能"""
    mse = mean_squared_error(actual_speeds, predicted_speeds)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_speeds, predicted_speeds)
    mae = np.mean(np.abs(actual_speeds - predicted_speeds))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }


def main():
    """主函数：运行速度预测示例"""
    setup_matplotlib_for_plotting()

    # 生成模拟数据
    print("正在生成无人车速度数据...")
    time_points, actual_speeds = generate_vehicle_data(num_points=50, base_speed=25, noise_level=1.5)

    # 创建预测器
    predictor = SimpleSpeedPredictor()

    # 方法1：移动平均预测
    print("\n使用移动平均方法进行预测...")
    predictor.set_method("moving_average", window_size=5)
    ma_predictions = predictor.predict(actual_speeds)
    ma_metrics = evaluate_prediction(actual_speeds, ma_predictions)

    # 方法2：线性回归预测
    print("使用线性回归方法进行预测...")
    predictor.set_method("linear_regression", history_size=8)
    lr_predictions = predictor.predict(actual_speeds)
    lr_metrics = evaluate_prediction(actual_speeds, lr_predictions)

    # 创建可视化图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 图1：移动平均预测结果
    ax1.plot(time_points, actual_speeds, 'b-', label='实际速度', linewidth=2, alpha=0.7)
    ax1.plot(time_points, ma_predictions, 'r--', label='移动平均预测', linewidth=2)
    ax1.set_title('移动平均预测方法', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('速度 (km/h)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 60)

    # 添加性能指标文本
    text1 = f'RMSE: {ma_metrics["RMSE"]:.2f}\nR²: {ma_metrics["R²"]:.3f}\nMAE: {ma_metrics["MAE"]:.2f}'
    ax1.text(0.02, 0.98, text1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 图2：线性回归预测结果
    ax2.plot(time_points, actual_speeds, 'b-', label='实际速度', linewidth=2, alpha=0.7)
    ax2.plot(time_points, lr_predictions, 'g--', label='线性回归预测', linewidth=2)
    ax2.set_title('线性回归预测方法', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间点')
    ax2.set_ylabel('速度 (km/h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 60)

    # 添加性能指标文本
    text2 = f'RMSE: {lr_metrics["RMSE"]:.2f}\nR²: {lr_metrics["R²"]:.3f}\nMAE: {lr_metrics["MAE"]:.2f}'
    ax2.text(0.02, 0.98, text2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 图3：预测误差对比
    ma_errors = actual_speeds - ma_predictions
    lr_errors = actual_speeds - lr_predictions

    ax3.plot(time_points, ma_errors, 'r-', label='移动平均误差', alpha=0.7)
    ax3.plot(time_points, lr_errors, 'g-', label='线性回归误差', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_title('预测误差对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('时间点')
    ax3.set_ylabel('误差 (km/h)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4：两种方法性能对比
    methods = ['移动平均', '线性回归']
    rmse_values = [ma_metrics['RMSE'], lr_metrics['RMSE']]
    mae_values = [ma_metrics['MAE'], lr_metrics['MAE']]

    x = np.arange(len(methods))
    width = 0.35

    ax4.bar(x - width / 2, rmse_values, width, label='RMSE', alpha=0.8)
    ax4.bar(x + width / 2, mae_values, width, label='MAE', alpha=0.8)

    ax4.set_title('预测性能对比', fontsize=14, fontweight='bold')
    ax4.set_xlabel('预测方法')
    ax4.set_ylabel('误差指标')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/autonomous_vehicle_speed_prediction_results.png', dpi=300, bbox_inches='tight')
    print(f"\n预测结果图表已保存到: autonomous_vehicle_speed_prediction_results.png")

    # 创建详细的时间序列对比图
    fig2, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.plot(time_points, actual_speeds, 'b-', label='实际速度', linewidth=3, alpha=0.8)
    ax.plot(time_points, ma_predictions, 'r--', label='移动平均预测', linewidth=2, marker='o', markersize=4)
    ax.plot(time_points, lr_predictions, 'g--', label='线性回归预测', linewidth=2, marker='s', markersize=4)

    ax.set_title('无人车速度预测完整对比图', fontsize=16, fontweight='bold')
    ax.set_xlabel('时间点', fontsize=12)
    ax.set_ylabel('速度 (km/h)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 60)

    # 添加统计信息
    info_text = f"""数据统计:
实际速度范围: {actual_speeds.min():.1f} - {actual_speeds.max():.1f} km/h
实际速度均值: {actual_speeds.mean():.1f} km/h

移动平均预测:
RMSE: {ma_metrics['RMSE']:.2f}, R²: {ma_metrics['R²']:.3f}

线性回归预测:
RMSE: {lr_metrics['RMSE']:.2f}, R²: {lr_metrics['R²']:.3f}"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/workspace/autonomous_vehicle_detailed_comparison.png', dpi=300, bbox_inches='tight')
    print(f"详细对比图已保存到: autonomous_vehicle_detailed_comparison.png")

    # 打印评估结果
    print("\n" + "=" * 50)
    print("预测性能评估结果")
    print("=" * 50)

    print(f"\n移动平均方法:")
    for metric, value in ma_metrics.items():
        print(f"  {metric}: {value:.3f}")

    print(f"\n线性回归方法:")
    for metric, value in lr_metrics.items():
        print(f"  {metric}: {value:.3f}")

    # 找出更好的方法
    if ma_metrics['RMSE'] < lr_metrics['RMSE']:
        print(f"\n结论: 移动平均方法在此数据集上表现更好 (RMSE更低)")
    else:
        print(f"\n结论: 线性回归方法在此数据集上表现更好 (RMSE更低)")

    return {
        'actual_speeds': actual_speeds,
        'ma_predictions': ma_predictions,
        'lr_predictions': lr_predictions,
        'time_points': time_points,
        'ma_metrics': ma_metrics,
        'lr_metrics': lr_metrics
    }


if __name__ == "__main__":
    results = main()