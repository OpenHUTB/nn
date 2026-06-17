"""
模型轻量化模块使用示例
演示如何优化ACC模型
"""

import numpy as np
import torch
import torch.nn as nn
from optimization import (
    ModelProfiler,
    ModelPruner,
    ModelQuantizer,
    KnowledgeDistiller,
    ModelOptimizer
)


# ACC策略网络
class ACCPolicyNetwork(nn.Module):
    """
    ACC系统的策略网络
    输入：观测状态（速度、距离等）
    输出：加速度动作
    """
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, x):
        return self.network(x)


def example_basic_profiling():
    """基本模型分析示例"""
    print("=" * 60)
    print("模型轻量化模块 - 基本分析示例")
    print("=" * 60)

    # 创建模型
    model = ACCPolicyNetwork()

    # 初始化分析器
    profiler = ModelProfiler()

    # 分析模型
    model_info = profiler.profile_model(model, "ACCPolicyNetwork")

    print(f"\n模型信息:")
    print(f"  名称: {model_info.name}")
    print(f"  参数量: {model_info.parameters:,}")
    print(f"  模型大小: {model_info.size_mb:.2f} MB")
    print(f"  层数: {model_info.layers}")

    # 测量推理时间
    input_shape = (1, 5)  # batch_size=1, state_dim=5
    inference_stats = profiler.measure_inference_time(model, input_shape)

    print(f"\n推理性能:")
    print(f"  平均时间: {inference_stats['mean_ms']:.3f} ms")
    print(f"  FPS: {inference_stats['fps']:.1f}")


def example_model_pruning():
    """模型剪枝示例"""
    print("\n" + "=" * 60)
    print("模型剪枝示例")
    print("=" * 60)

    profiler = ModelProfiler()
    pruner = ModelPruner(pruning_ratio=0.4)

    # 创建原始模型
    original_model = ACCPolicyNetwork()

    # 分析原始模型
    original_info = profiler.profile_model(original_model, "Original")
    print(f"\n原始模型大小: {original_info.size_mb:.2f} MB")
    print(f"原始参数量: {original_info.parameters:,}")

    # 执行剪枝
    print("\n执行权重剪枝（40%稀疏度）...")
    pruned_model = pruner.weight_pruning(original_model)

    # 分析剪枝效果
    pruning_stats = pruner.get_pruning_statistics(pruned_model)
    print(f"\n剪枝结果:")
    print(f"  稀疏度: {pruning_stats['sparsity_percent']:.2f}%")
    print(f"  非零参数: {pruning_stats['remaining_params']:,}")

    # 推理时间对比
    input_shape = (1, 5)
    original_time = profiler.measure_inference_time(original_model, input_shape)
    pruned_time = profiler.measure_inference_time(pruned_model, input_shape)

    speedup = original_time['mean_ms'] / pruned_time['mean_ms']
    print(f"\n性能提升:")
    print(f"  加速比: {speedup:.2f}x")


def example_model_quantization():
    """模型量化示例"""
    print("\n" + "=" * 60)
    print("模型量化示例")
    print("=" * 60)

    profiler = ModelProfiler()
    quantizer = ModelQuantizer(quantization_type="dynamic")

    # 创建模型
    original_model = ACCPolicyNetwork()

    # 分析原始模型
    original_info = profiler.profile_model(original_model, "Original")
    print(f"\n原始模型大小: {original_info.size_mb:.2f} MB")

    # 执行动态量化
    print("\n执行INT8动态量化...")
    quantized_model = quantizer.quantize_model(original_model)

    # 分析量化模型
    quantized_info = profiler.profile_model(quantized_model, "Quantized")
    print(f"\n量化模型大小: {quantized_info.size_mb:.2f} MB")

    # 计算压缩效果
    compression_ratio = original_info.size_mb / quantized_info.size_mb
    size_reduction = (1 - quantized_info.size_mb / original_info.size_mb) * 100

    print(f"\n压缩效果:")
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  大小减少: {size_reduction:.2f}%")

    # 推理时间对比
    input_shape = (1, 5)
    original_time = profiler.measure_inference_time(original_model, input_shape)
    quantized_time = profiler.measure_inference_time(quantized_model, input_shape)

    speedup = original_time['mean_ms'] / quantized_time['mean_ms']
    print(f"\n性能提升:")
    print(f"  加速比: {speedup:.2f}x")


def example_knowledge_distillation():
    """知识蒸馏示例"""
    print("\n" + "=" * 60)
    print("知识蒸馏示例")
    print("=" * 60)

    profiler = ModelProfiler()

    # 创建教师和学生模型
    teacher_model = ACCPolicyNetwork(hidden_dim=256)  # 大模型
    student_model = ACCPolicyNetwork(hidden_dim=64)   # 小模型

    # 分析模型大小
    teacher_info = profiler.profile_model(teacher_model, "Teacher")
    student_info = profiler.profile_model(student_model, "Student")

    print(f"\n教师模型:")
    print(f"  参数量: {teacher_info.parameters:,}")
    print(f"  大小: {teacher_info.size_mb:.2f} MB")

    print(f"\n学生模型:")
    print(f"  参数量: {student_info.parameters:,}")
    print(f"  大小: {student_info.size_mb:.2f} MB")

    # 初始化蒸馏器
    distiller = KnowledgeDistiller(temperature=3.0, alpha=0.7)

    print(f"\n蒸馏配置:")
    print(f"  温度: {distiller.temperature}")
    print(f"  软标签权重: {distiller.alpha}")

    # 模拟蒸馏过程
    print("\n模拟蒸馏训练...")
    num_samples = 100
    dummy_data = torch.randn(num_samples, 5)
    dummy_labels = torch.randn(num_samples, 1)

    teacher_output = teacher_model(dummy_data)
    student_output = student_model(dummy_data)
    loss = distiller.distillation_loss(student_output, teacher_output, dummy_labels)

    print(f"  蒸馏损失: {loss.item():.4f}")
    print("\n蒸馏完成！学生模型已学习教师模型的知识。")


def example_comprehensive_optimization():
    """综合优化示例"""
    print("\n" + "=" * 60)
    print("综合优化示例")
    print("=" * 60)

    optimizer = ModelOptimizer()
    profiler = ModelProfiler()

    # 创建模型
    model = ACCPolicyNetwork()

    # 分析原始模型
    original_info = profiler.profile_model(model, "Original")
    print(f"\n原始模型:")
    print(f"  参数量: {original_info.parameters:,}")
    print(f"  大小: {original_info.size_mb:.2f} MB")

    # 优化配置
    config = {
        'pruning': True,
        'pruning_ratio': 0.3,
        'quantization': True,
        'quantization_type': 'dynamic'
    }

    print(f"\n优化策略:")
    print(f"  剪枝: 启用 (比例: {config['pruning_ratio']})")
    print(f"  量化: 启用 (类型: {config['quantization_type']})")

    # 执行优化
    print("\n执行优化流水线...")
    optimized_model, result = optimizer.optimize_pipeline(model, config)

    # 打印结果
    print(f"\n优化结果:")
    print(f"  原始大小: {result.original_size:.2f} MB")
    print(f"  优化大小: {result.optimized_size:.2f} MB")
    print(f"  压缩比: {result.compression_ratio:.2f}x")
    print(f"  大小减少: {(1 - result.optimized_size / result.original_size) * 100:.2f}%")

    # 推理性能对比
    input_shape = (1, 5)
    original_time = profiler.measure_inference_time(model, input_shape)
    optimized_time = profiler.measure_inference_time(optimized_model, input_shape)

    speedup = original_time['mean_ms'] / optimized_time['mean_ms']
    print(f"\n推理性能:")
    print(f"  原始时间: {original_time['mean_ms']:.3f} ms")
    print(f"  优化时间: {optimized_time['mean_ms']:.3f} ms")
    print(f"  加速比: {speedup:.2f}x")


def example_save_optimized_model():
    """保存优化模型示例"""
    print("\n" + "=" * 60)
    print("保存优化模型示例")
    print("=" * 60)

    optimizer = ModelOptimizer()

    # 创建并优化模型
    model = ACCPolicyNetwork()
    config = {
        'pruning': True,
        'pruning_ratio': 0.3,
        'quantization': True,
        'quantization_type': 'dynamic'
    }

    optimized_model, _ = optimizer.optimize_pipeline(model, config)

    # 保存模型
    save_path = "optimized_acc_model.pt"
    optimizer.save_optimized_model(optimized_model, save_path, format="pytorch")

    print(f"\n模型已保存到: {save_path}")
    print("保存格式: PyTorch state_dict")


def example_real_world_usage():
    """实际应用示例"""
    print("\n" + "=" * 60)
    print("实际应用示例 - ACC系统优化")
    print("=" * 60)

    profiler = ModelProfiler()
    optimizer = ModelOptimizer()

    # 模拟ACC系统场景
    print("\n场景: 嵌入式ACC系统")
    print("需求: 模型大小 < 1MB, 推理时间 < 10ms")

    # 创建原始模型
    model = ACCPolicyNetwork(hidden_dim=128)

    # 分析原始模型
    original_info = profiler.profile_model(model, "Original")
    original_time = profiler.measure_inference_time(model, (1, 5))

    print(f"\n原始模型:")
    print(f"  大小: {original_info.size_mb:.2f} MB")
    print(f"  推理时间: {original_time['mean_ms']:.3f} ms")

    # 判断是否满足需求
    meets_size = original_info.size_mb < 1.0
    meets_time = original_time['mean_ms'] < 10.0

    print(f"\n需求检查:")
    print(f"  大小需求: {'✓' if meets_size else '✗'} ({original_info.size_mb:.2f} MB < 1 MB)")
    print(f"  时间需求: {'✓' if meets_time else '✗'} ({original_time['mean_ms']:.3f} ms < 10 ms)")

    if not (meets_size and meets_time):
        print("\n需要优化模型以满足需求...")

        # 优化配置
        config = {
            'pruning': True,
            'pruning_ratio': 0.5,
            'quantization': True,
            'quantization_type': 'dynamic'
        }

        # 执行优化
        optimized_model, result = optimizer.optimize_pipeline(model, config)
        optimized_time = profiler.measure_inference_time(optimized_model, (1, 5))

        print(f"\n优化后模型:")
        print(f"  大小: {result.optimized_size:.2f} MB")
        print(f"  推理时间: {optimized_time['mean_ms']:.3f} ms")

        # 再次检查需求
        meets_size = result.optimized_size < 1.0
        meets_time = optimized_time['mean_ms'] < 10.0

        print(f"\n优化后需求检查:")
        print(f"  大小需求: {'✓' if meets_size else '✗'} ({result.optimized_size:.2f} MB < 1 MB)")
        print(f"  时间需求: {'✓' if meets_time else '✗'} ({optimized_time['mean_ms']:.3f} ms < 10 ms)")

        if meets_size and meets_time:
            print("\n✓ 模型已满足嵌入式系统需求！")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_profiling()
    example_model_pruning()
    example_model_quantization()
    example_knowledge_distillation()
    example_comprehensive_optimization()
    example_save_optimized_model()
    example_real_world_usage()

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)