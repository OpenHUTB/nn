import torch
import torch.nn as nn
from .routers import *
from .experts import *
from .loss import MoELoss

class ESMoE(nn.Module):
    """Improved MoE with pluggable routers/experts and a shared expert for stability."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = 'simple',   # ['simple', 'ghost', 'inverted']
        router_type: str = 'efficient',# ['efficient', 'local', 'adaptive']
        noise_std: float = 1.0,
        balance_loss_coeff: float = 0.01,
        router_z_loss_coeff: float = 1e-3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        
        # 1) Instantiate Router
        if router_type == 'local':
            self.routing = LocalRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        elif router_type == 'adaptive':
            self.routing = AdaptiveRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        else:
            self.routing = EfficientSpatialRouter(in_channels, num_experts, top_k=top_k, noise_std=noise_std)

        # 2) Instantiate Experts
        self.experts = nn.ModuleList()
        if expert_type == 'ghost':
            expert_cls = GhostExpert
        elif expert_type == 'inverted':
            expert_cls = InvertedResidualExpert
        else:
            expert_cls = SimpleExpert
            
        for _ in range(num_experts):
            self.experts.append(expert_cls(in_channels, out_channels))
            
        # 3) Shared expert (Always active)
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self._init_weights()
        self.aux_loss = torch.tensor(0.0)
        self.moe_loss_fn = MoELoss(balance_loss_coeff, router_z_loss_coeff, num_experts, top_k)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Robust router init: find the last Conv layer to initialize
        # Keep initial expert probabilities nearly uniform
        for m in self.routing.router.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            nn.init.normal_(last_conv.weight, mean=0, std=0.01)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1) Routing (standardized interface)
        # loss_dict contains training loss inputs; empty during inference
        routing_weights, routing_indices, loss_dict = self.routing(x)
        
        # 2) Shared expert compute
        shared_out = self.shared_expert(x)
        
        # 3) Sparse expert compute
        # Initialize outputs with zeros
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        
        indices_flat = routing_indices.view(B, self.top_k)
        weights_flat = routing_weights.view(B, self.top_k)
        
        for i in range(self.num_experts):
            # Find all samples assigned to expert i
            mask = (indices_flat == i)
            if mask.any():
                batch_idx, k_idx = torch.where(mask)
                
                # Select input and compute
                inp = x[batch_idx]
                out = self.experts[i](inp)
                
                # Select weights and broadcast
                w = weights_flat[batch_idx, k_idx].view(-1, 1, 1, 1)
                
                # Accumulate results
                expert_output.index_add_(0, batch_idx, out.to(expert_output.dtype) * w.to(expert_output.dtype))
        
        final_output = shared_out + expert_output

        # 4) Compute and return Loss during training
        if self.training and loss_dict:
             self.aux_loss = self.moe_loss_fn(loss_dict['router_probs'], loss_dict['router_logits'], loss_dict['topk_indices'])
        else:
             pass

        return final_output