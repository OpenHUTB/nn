"""
模型轻量化模块测试脚本
验证剪枝、量化、蒸馏等功能
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimization import (
    ModelProfiler,
    ModelPruner,
    ModelQuantizer,
    KnowledgeDistiller,
    ModelOptimizer,
    PruningConfig,
    QuantizationConfig,
    DistillationConfig,
    OptimizationConfig
)


class SimpleACCModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class LargeACCModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def test_model_profiler():
    print("=" * 60)
    print("模型分析器测试")
    print("=" * 60)

    profiler = ModelProfiler()
    model = SimpleACCModel()

    model_info = profiler.profile_model(model, "SimpleACCModel", input_shape=(1, 5))
    profiler.print_profile(model_info)

    input_shape = (1, 5)
    inference_stats = profiler.measure_inference_time(model, input_shape, num_runs=100)

    print("\n推理时间统计:")
    print(f"  平均时间: {inference_stats['mean_ms']:.4f} ms")
    print(f"  标准差: {inference_stats['std_ms']:.4f} ms")
    print(f"  FPS: {inference_stats['fps']:.1f}")

    return profiler, model_info


def test_model_pruning():
    print("\n" + "=" * 60)
    print("模型剪枝测试")
    print("=" * 60)

    profiler = ModelProfiler()

    pruning_config = PruningConfig(method="weight", ratio=0.3)
    pruner = ModelPruner(pruning_config)

    original_model = SimpleACCModel()
    original_info = profiler.profile_model(original_model, "Original")

    print("\n原始模型:")
    profiler.print_profile(original_info)

    pruned_model = pruner.prune(original_model)
    pruning_stats = pruner.get_pruning_statistics(pruned_model)

    print("\n剪枝统计:")
    print(f"  总参数: {pruning_stats['total_params']:,}")
    print(f"  零参数: {pruning_stats['zero_params']:,}")
    print(f"  稀疏度: {pruning_stats['sparsity_percent']:.2f}%")

    return pruner, pruning_stats


def test_model_quantization():
    print("\n" + "=" * 60)
    print("模型量化测试")
    print("=" * 60)

    profiler = ModelProfiler()

    quantization_config = QuantizationConfig(type="dynamic")
    quantizer = ModelQuantizer(quantization_config)

    original_model = SimpleACCModel()
    original_info = profiler.profile_model(original_model, "Original")

    print("\n原始模型:")
    profiler.print_profile(original_info)

    quantized_model = quantizer.quantize_model(original_model)
    quantized_info = profiler.profile_model(quantized_model, "Quantized")

    print("\n量化模型:")
    profiler.print_profile(quantized_info)

    compression_ratio = original_info.size_mb / max(quantized_info.size_mb, 0.001)
    size_reduction = (1 - quantized_info.size_mb / max(original_info.size_mb, 0.001)) * 100

    print("\n量化效果:")
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  大小减少: {size_reduction:.2f}%")

    return quantizer, quantized_info


def test_knowledge_distillation():
    print("\n" + "=" * 60)
    print("知识蒸馏测试")
    print("=" * 60)

    profiler = ModelProfiler()

    distillation_config = DistillationConfig(temperature=3.0, alpha=0.7)
    distiller = KnowledgeDistiller(distillation_config)

    teacher_model = LargeACCModel()
    student_model = SimpleACCModel()

    teacher_info = profiler.profile_model(teacher_model, "Teacher")
    student_info = profiler.profile_model(student_model, "Student")

    print("\n教师模型:")
    profiler.print_profile(teacher_info)
    print("\n学生模型:")
    profiler.print_profile(student_info)

    num_samples = 100
    train_data = torch.randn(num_samples, 5)
    train_labels = torch.randn(num_samples, 1)
    train_loader = [(train_data[i:i+10], train_labels[i:i+10]) for i in range(0, num_samples, 10)]

    print("\n模拟蒸馏训练...")
    for epoch in range(3):
        total_loss = 0.0
        for data, labels in train_loader:
            teacher_output = teacher_model(data)
            student_output = student_model(data)
            loss = distiller.distillation_loss(student_output, teacher_output, labels)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    print("\n蒸馏完成！")
    return distiller


def test_optimization_pipeline():
    print("\n" + "=" * 60)
    print("优化流水线测试")
    print("=" * 60)

    optimizer = ModelOptimizer()
    profiler = ModelProfiler()

    model = SimpleACCModel()
    input_shape = (1, 5)

    config = OptimizationConfig(
        pruning=PruningConfig(enabled=True, method="weight", ratio=0.3),
        quantization=QuantizationConfig(enabled=True, type="dynamic"),
        distillation=DistillationConfig(enabled=False)
    )

    print("\n优化配置:")
    print(f"  剪枝: {config.pruning.enabled} ({config.pruning.method}, ratio={config.pruning.ratio})")
    print(f"  量化: {config.quantization.enabled} ({config.quantization.type})")
    print(f"  蒸馏: {config.distillation.enabled}")

    optimizer.config = config
    optimized_model, result = optimizer.optimize_pipeline(model, input_shape)

    print("\n优化结果:")
    print(f"  原始大小: {result.original_size:.4f} MB")
    print(f"  优化大小: {result.optimized_size:.4f} MB")
    print(f"  压缩比: {result.compression_ratio:.2f}x")
    print(f"  加速比: {result.speedup:.2f}x")
    print(f"  参数减少: {result.params_reduction:,}")

    return optimizer, result


def visualize_optimization_results():
    profiler = ModelProfiler()

    original = SimpleACCModel()
    pruned_config = PruningConfig(ratio=0.3)
    pruned_model = ModelPruner(pruned_config).prune(SimpleACCModel())
    quantized_model = ModelQuantizer().quantize_model(SimpleACCModel())

    original_info = profiler.profile_model(original, "Original")
    pruned_info = profiler.profile_model(pruned_model, "Pruned")
    quantized_info = profiler.profile_model(quantized_model, "Quantized")

    input_shape = (1, 5)
    original_time = profiler.measure_inference_time(original, input_shape)
    pruned_time = profiler.measure_inference_time(pruned_model, input_shape)
    quantized_time = profiler.measure_inference_time(quantized_model, input_shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['Original', 'Pruned', 'Quantized']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    ax1 = axes[0]
    sizes = [original_info.size_mb, pruned_info.size_mb, quantized_info.size_mb]
    bars1 = ax1.bar(models, sizes, color=colors)
    ax1.set_ylabel('Size (MB)')
    ax1.set_title('Model Size Comparison')
    ax1.grid(True, alpha=0.3)
    for bar, size in zip(bars1, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{size:.4f}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    params = [original_info.parameters, pruned_info.parameters, quantized_info.parameters]
    bars2 = ax2.bar(models, params, color=colors)
    ax2.set_ylabel('Parameters')
    ax2.set_title('Parameter Count Comparison')
    ax2.grid(True, alpha=0.3)
    for bar, param in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{param:,}', ha='center', va='bottom', fontsize=8)

    ax3 = axes[2]
    times = [original_time['mean_ms'], pruned_time['mean_ms'], quantized_time['mean_ms']]
    bars3 = ax3.bar(models, times, color=colors)
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Inference Time Comparison')
    ax3.grid(True, alpha=0.3)
    for bar, time in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{time:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
    print("\n可视化结果已保存到 optimization_comparison.png")


def comprehensive_comparison():
    print("\n" + "=" * 60)
    print("综合对比测试")
    print("=" * 60)

    profiler = ModelProfiler()

    models = {
        'Small': SimpleACCModel(hidden_dim=32),
        'Medium': SimpleACCModel(hidden_dim=64),
        'Large': LargeACCModel(),
    }

    print("\n模型对比:")
    print("-" * 60)
    print(f"{'Model':<10} {'Params':<12} {'Size (MB)':<12} {'Layers':<8} {'FLOPs':<15}")
    print("-" * 60)

    for name, model in models.items():
        info = profiler.profile_model(model, name, input_shape=(1, 5))
        flops_str = f"{info.flops:,}" if info.flops else "-"
        print(f"{name:<10} {info.parameters:<12,} {info.size_mb:<12.4f} {info.layers:<8} {flops_str:<15}")

    input_shape = (1, 5)
    print("\n推理性能对比:")
    print("-" * 60)
    print(f"{'Model':<10} {'Time (ms)':<12} {'FPS':<10}")
    print("-" * 60)

    for name, model in models.items():
        stats = profiler.measure_inference_time(model, input_shape)
        print(f"{name:<10} {stats['mean_ms']:<12.4f} {stats['fps']:<10.1f}")


if __name__ == "__main__":
    test_model_profiler()
    test_model_pruning()
    test_model_quantization()
    test_knowledge_distillation()
    test_optimization_pipeline()
    visualize_optimization_results()
    comprehensive_comparison()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)