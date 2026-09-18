"""Historical direct pre-RoPE MLP projector used by raw Concat checkpoints."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from rosetta.model.projector import Projector
from rosetta.utils.registry import capture_init_args, register_model


@register_model
@capture_init_args
class DirectPreRopeMLPProjector(Projector):
    def __init__(self, sharer_num_kv_heads: int, sharer_head_dim: int,
                 receiver_num_kv_heads: int, receiver_head_dim: int,
                 hidden_dim: int = 1024, activation: str = "gelu",
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.sharer_num_kv_heads, self.sharer_head_dim = int(sharer_num_kv_heads), int(sharer_head_dim)
        self.receiver_num_kv_heads, self.receiver_head_dim = int(receiver_num_kv_heads), int(receiver_head_dim)
        self.hidden_dim, self.activation = int(hidden_dim), str(activation).lower()
        activation_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(self.activation)
        if activation_cls is None:
            raise ValueError("Direct pre-RoPE MLP activation must be gelu, relu, or silu.")
        source = self.sharer_num_kv_heads * self.sharer_head_dim
        target = self.receiver_num_kv_heads * self.receiver_head_dim
        self.key_mlp = nn.Sequential(nn.Linear(source, self.hidden_dim, dtype=dtype), activation_cls(), nn.Linear(self.hidden_dim, target, dtype=dtype))
        self.value_mlp = nn.Sequential(nn.Linear(source, self.hidden_dim, dtype=dtype), activation_cls(), nn.Linear(self.hidden_dim, target, dtype=dtype))

    def project(self, source_kv: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        key, value = source_kv
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("Direct pre-RoPE MLP expects matching Sharer K/V [B,H,S,D].")
        batch, heads, length, dim = key.shape
        if (heads, dim) != (self.sharer_num_kv_heads, self.sharer_head_dim):
            raise ValueError("Sharer KV geometry does not match the direct MLP.")
        def project_one(values: Tensor, mlp: nn.Module) -> Tensor:
            channels = values.transpose(1, 2).contiguous().reshape(batch, length, -1)
            output = mlp(channels.to(dtype=mlp[0].weight.dtype))
            return output.reshape(batch, length, self.receiver_num_kv_heads, self.receiver_head_dim).transpose(1, 2).contiguous()
        return project_one(key, self.key_mlp), project_one(value, self.value_mlp)

    def forward(self, source_kv: tuple[Tensor, Tensor], target_kv=None) -> tuple[Tensor, Tensor]:
        del target_kv
        return self.project(source_kv)
