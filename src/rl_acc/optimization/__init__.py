"""
模型轻量化模块 - Model Optimization Module
实现模型压缩、量化、剪枝等轻量化技术
用于减小模型体积、加速推理、降低内存占用
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
import os


@dataclass
class ModelInfo:
    name: str
    parameters: int
    trainable_params: int
    size_mb: float
    layers: int
    flops: Optional[int] = None
    memory_usage_mb: Optional[float] = None


@dataclass
class OptimizationResult:
    original_size: float
    optimized_size: float
    compression_ratio: float
    speedup: float
    accuracy_loss: float
    method: str
    params_reduction: int = 0
    flops_reduction: Optional[int] = None


@dataclass
class PruningConfig:
    enabled: bool = True
    ratio: float = 0.3
    threshold: Optional[float] = None
    method: str = "weight"
    layer_ratios: Dict[str, float] = field(default_factory=dict)


@dataclass
class QuantizationConfig:
    enabled: bool = True
    type: str = "dynamic"
    dtype: torch.dtype = torch.qint8
    backend: str = "fbgemm"


@dataclass
class DistillationConfig:
    enabled: bool = False
    temperature: float = 3.0
    alpha: float = 0.7
    epochs: int = 10
    learning_rate: float = 0.001


@dataclass
class OptimizationConfig:
    pruning: PruningConfig = field(default_factory=PruningConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)


class FLOPCalculator:
    @staticmethod
    def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """计算模型的浮点运算次数"""
        flops = 0
        dummy_input = torch.randn(*input_shape)

        def hook(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Linear):
                in_features = input[0].shape[-1]
                out_features = output.shape[-1]
                flops += in_features * out_features * input[0].shape[0]
            elif isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0]
                output_size = output.shape[2] * output.shape[3]
                flops += in_channels * out_channels * kernel_size * kernel_size * output_size * input[0].shape[0]

        handles = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handles.append(module.register_forward_hook(hook))

        with torch.no_grad():
            model(dummy_input)

        for h in handles:
            h.remove()

        return flops


class ModelProfiler:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.flop_calculator = FLOPCalculator()

    def profile_model(self, model: nn.Module, model_name: str = "model",
                      input_shape: Optional[Tuple[int, ...]] = None) -> ModelInfo:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = (param_size + buffer_size) / (1024 * 1024)

        num_layers = len(list(model.modules()))

        flops = None
        if input_shape is not None:
            flops = self.flop_calculator.calculate_flops(model, input_shape)

        return ModelInfo(
            name=model_name,
            parameters=total_params,
            trainable_params=trainable_params,
            size_mb=total_size,
            layers=num_layers,
            flops=flops
        )

    def measure_inference_time(self, model: nn.Module, input_shape: Tuple[int, ...],
                              num_runs: int = 100, device: str = "cpu") -> Dict[str, float]:
        model = model.to(device)
        model.eval()

        dummy_input = torch.randn(*input_shape).to(device)

        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append(end - start)

        mean_time = np.mean(times)
        std_time = np.std(times)

        return {
            'mean_ms': mean_time * 1000,
            'std_ms': std_time * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / mean_time,
            'latency_ms': mean_time * 1000
        }

    def compare_models(self, original: nn.Module, optimized: nn.Module,
                       input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        original_info = self.profile_model(original, "original", input_shape)
        optimized_info = self.profile_model(optimized, "optimized", input_shape)

        original_time = self.measure_inference_time(original, input_shape)
        optimized_time = self.measure_inference_time(optimized, input_shape)

        size_reduction = (1 - optimized_info.size_mb / max(original_info.size_mb, 0.001)) * 100
        speedup = original_time['mean_ms'] / max(optimized_time['mean_ms'], 0.001)

        return {
            'original': {
                'params': original_info.parameters,
                'size_mb': original_info.size_mb,
                'flops': original_info.flops,
                'inference_ms': original_time['mean_ms'],
                'fps': original_time['fps']
            },
            'optimized': {
                'params': optimized_info.parameters,
                'size_mb': optimized_info.size_mb,
                'flops': optimized_info.flops,
                'inference_ms': optimized_time['mean_ms'],
                'fps': optimized_time['fps']
            },
            'improvement': {
                'size_reduction_percent': size_reduction,
                'speedup': speedup,
                'params_reduction': original_info.parameters - optimized_info.parameters,
                'flops_reduction': (original_info.flops - optimized_info.flops) if original_info.flops else None
            }
        }

    def print_profile(self, model_info: ModelInfo):
        print("\n" + "=" * 60)
        print(f"模型分析: {model_info.name}")
        print("=" * 60)
        print(f"总参数量: {model_info.parameters:,}")
        print(f"可训练参数: {model_info.trainable_params:,}")
        print(f"模型大小: {model_info.size_mb:.4f} MB")
        print(f"层数: {model_info.layers}")
        if model_info.flops:
            print(f"FLOPs: {model_info.flops:,}")
        print("=" * 60)


class ModelPruner:
    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()

    def weight_pruning(self, model: nn.Module, threshold: Optional[float] = None) -> nn.Module:
        pruned_model = model

        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if threshold is None:
                    ratio = self.config.layer_ratios.get(name, self.config.ratio)
                    prune.l1_unstructured(module, name='weight', amount=ratio)
                else:
                    mask = torch.abs(module.weight.data) > threshold
                    prune.custom_from_mask(module, name='weight', mask=mask.float())

        return pruned_model

    def structured_pruning(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                ratio = self.config.layer_ratios.get(name, self.config.ratio)
                num_neurons = module.out_features
                num_to_prune = int(num_neurons * ratio)

                importance = torch.sum(torch.abs(module.weight.data), dim=1)
                _, indices = torch.sort(importance)
                prune_indices = indices[:num_to_prune]

                keep_mask = torch.ones(module.weight.data.shape[0], dtype=torch.bool)
                keep_mask[prune_indices] = False
                prune.custom_from_mask(module, name='weight', mask=keep_mask.unsqueeze(1).expand_as(module.weight.data))

                if module.bias is not None:
                    bias_mask = keep_mask.clone()
                    prune.custom_from_mask(module, name='bias', mask=bias_mask)

            elif isinstance(module, nn.Conv2d):
                ratio = self.config.layer_ratios.get(name, self.config.ratio)
                num_filters = module.out_channels
                num_to_prune = int(num_filters * ratio)

                importance = torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3))
                _, indices = torch.sort(importance)
                prune_indices = indices[:num_to_prune]

                keep_mask = torch.ones(module.weight.data.shape[0], dtype=torch.bool)
                keep_mask[prune_indices] = False
                prune.custom_from_mask(module, name='weight', mask=keep_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(module.weight.data))

        return model

    def channel_pruning(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                ratio = self.config.layer_ratios.get(name, self.config.ratio)
                num_channels = module.in_channels
                num_to_prune = int(num_channels * ratio)

                importance = torch.sum(torch.abs(module.weight.data), dim=(0, 2, 3))
                _, indices = torch.sort(importance)
                prune_indices = indices[:num_to_prune]

                keep_mask = torch.ones(module.weight.data.shape[1], dtype=torch.bool)
                keep_mask[prune_indices] = False
                prune.custom_from_mask(module, name='weight', mask=keep_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(module.weight.data))

        return model

    def prune(self, model: nn.Module) -> nn.Module:
        if self.config.method == "weight":
            return self.weight_pruning(model, self.config.threshold)
        elif self.config.method == "structured":
            return self.structured_pruning(model)
        elif self.config.method == "channel":
            return self.channel_pruning(model)
        else:
            raise ValueError(f"Unknown pruning method: {self.config.method}")

    def remove_pruning(self, model: nn.Module) -> nn.Module:
        for module in model.modules():
            try:
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')
            except (AttributeError, ValueError):
                continue
        return model

    def get_pruning_statistics(self, model: nn.Module) -> Dict[str, float]:
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

        sparsity = zero_params / max(total_params, 1) * 100

        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'remaining_params': total_params - zero_params,
            'sparsity_percent': sparsity
        }


class ModelQuantizer:
    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def dynamic_quantization(self, model: nn.Module) -> nn.Module:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=self.config.dtype
        )
        return quantized_model

    def static_quantization(self, model: nn.Module,
                           calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(self.config.backend)
        prepared_model = torch.quantization.prepare(model)

        if calibration_data is not None:
            with torch.no_grad():
                if isinstance(calibration_data, list):
                    for data in calibration_data:
                        prepared_model(data)
                else:
                    prepared_model(calibration_data)

        quantized_model = torch.quantization.convert(prepared_model)
        return quantized_model

    def quantize_model(self, model: nn.Module,
                      calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        if self.config.type == "dynamic":
            return self.dynamic_quantization(model)
        elif self.config.type == "static":
            return self.static_quantization(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization type: {self.config.type}")


class KnowledgeDistiller:
    def __init__(self, config: Optional[DistillationConfig] = None):
        self.config = config or DistillationConfig()

    def distillation_loss(self, student_output: torch.Tensor,
                         teacher_output: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        soft_teacher = torch.softmax(teacher_output / self.config.temperature, dim=1)
        soft_student = torch.log_softmax(student_output / self.config.temperature, dim=1)
        soft_loss = torch.nn.functional.kl_div(
            soft_student, soft_teacher, reduction='batchmean'
        ) * (self.config.temperature ** 2)

        if labels.dim() > 1:
            hard_loss = torch.nn.functional.mse_loss(student_output, labels)
        else:
            hard_loss = torch.nn.functional.cross_entropy(student_output, labels)

        total_loss = self.config.alpha * soft_loss + (1 - self.config.alpha) * hard_loss

        return total_loss

    def intermediate_distillation_loss(self, student_features: List[torch.Tensor],
                                      teacher_features: List[torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            total_loss += torch.nn.functional.mse_loss(s_feat, t_feat.detach())
        return total_loss / len(student_features)

    def distill(self, teacher_model: nn.Module, student_model: nn.Module,
               train_loader: Any, epochs: int = None,
               learning_rate: float = None) -> nn.Module:
        epochs = epochs or self.config.epochs
        learning_rate = learning_rate or self.config.learning_rate

        teacher_model.eval()
        student_model.train()

        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for data, labels in train_loader:
                with torch.no_grad():
                    teacher_output = teacher_model(data)

                student_output = student_model(data)
                loss = self.distillation_loss(student_output, teacher_output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        return student_model


class ModelOptimizer:
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.profiler = ModelProfiler()
        self.pruner = ModelPruner(self.config.pruning)
        self.quantizer = ModelQuantizer(self.config.quantization)
        self.distiller = KnowledgeDistiller(self.config.distillation)

    def optimize_pipeline(self, model: nn.Module,
                         input_shape: Tuple[int, ...],
                         calibration_data: Optional[torch.Tensor] = None,
                         distillation_loader: Optional[Any] = None,
                         teacher_model: Optional[nn.Module] = None) -> Tuple[nn.Module, OptimizationResult]:
        original_info = self.profiler.profile_model(model, "original", input_shape)

        optimized_model = model

        if self.config.distillation.enabled and teacher_model and distillation_loader:
            optimized_model = self.distiller.distill(
                teacher_model, optimized_model, distillation_loader
            )
            print(f"知识蒸馏完成，epochs: {self.config.distillation.epochs}")

        if self.config.pruning.enabled:
            optimized_model = self.pruner.prune(optimized_model)
            optimized_model = self.pruner.remove_pruning(optimized_model)
            print(f"剪枝完成，方法: {self.config.pruning.method}, 比例: {self.config.pruning.ratio}")

        if self.config.quantization.enabled:
            optimized_model = self.quantizer.quantize_model(optimized_model, calibration_data)
            print(f"量化完成，类型: {self.config.quantization.type}")

        optimized_info = self.profiler.profile_model(optimized_model, "optimized", input_shape)

        compression_ratio = original_info.size_mb / max(optimized_info.size_mb, 0.001)

        original_time = self.profiler.measure_inference_time(model, input_shape)
        optimized_time = self.profiler.measure_inference_time(optimized_model, input_shape)
        speedup = original_time['mean_ms'] / max(optimized_time['mean_ms'], 0.001)

        result = OptimizationResult(
            original_size=original_info.size_mb,
            optimized_size=optimized_info.size_mb,
            compression_ratio=compression_ratio,
            speedup=speedup,
            accuracy_loss=0.0,
            method=f"pruning={self.config.pruning.enabled},quantization={self.config.quantization.enabled},distillation={self.config.distillation.enabled}",
            params_reduction=original_info.parameters - optimized_info.parameters
        )

        return optimized_model, result

    def save_optimized_model(self, model: nn.Module, path: str,
                            format: str = "pytorch"):
        if format == "pytorch":
            torch.save(model.state_dict(), path)
        elif format == "torchscript":
            scripted_model = torch.jit.script(model)
            scripted_model.save(path)
        elif format == "onnx":
            dummy_input = torch.randn(1, 5)
            torch.onnx.export(model, dummy_input, path)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"模型已保存到: {path}")

    def load_optimized_model(self, model: nn.Module, path: str) -> nn.Module:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        print(f"模型已从: {path} 加载")
        return model