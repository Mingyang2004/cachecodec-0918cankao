"""新版 CacheCodec Joint Cache Bottleneck (JCB).

The module is intentionally independent of the collaboration operator.  It
encodes one Sharer layer's K/V jointly, exposes two 64-wide transport views,
and decodes them into the requested receiver geometry for Concat/Prefix use.
"""

from __future__ import annotations

import torch

from rosetta.model.lcf_projected_kv import LCFProjectedKVProjector
from rosetta.utils.registry import capture_init_args, register_model
from torch import nn


@register_model
@capture_init_args
class JCBProjector(LCFProjectedKVProjector):
    """JCB encoder and independent K/V decoder branches.

    The implementation shares the proven LCFProjectedKV tensor contract so
    existing checkpoints can be loaded, while the public class name matches
    the revised CacheCodec design.
    """

    def __init__(
        self,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        shared_latent_dim: int = 128,
        key_latent_dim: int = 64,
        value_latent_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if shared_latent_dim != 128 or key_latent_dim != 64 or value_latent_dim != 64:
            raise ValueError(
                "JCB uses the fixed revised dimensions shared=128, key=64, value=64."
            )
        super().__init__(
            sharer_num_kv_heads=sharer_num_kv_heads,
            sharer_head_dim=sharer_head_dim,
            receiver_num_kv_heads=receiver_num_kv_heads,
            receiver_head_dim=receiver_head_dim,
            shared_latent_dim=shared_latent_dim,
            key_latent_dim=key_latent_dim,
            value_latent_dim=value_latent_dim,
            dtype=dtype,
        )


@register_model
@capture_init_args
class JCBC2CProjector(JCBProjector):
    """JCB transport adapter for C2C's original fusion boundary.

    Unlike :class:`JCBProjector`, this branch reconstructs the Sharer geometry
    and never consumes Receiver KV.  The returned cache can therefore be fed
    into the unchanged C2C projector/fuser.
    """

    def __init__(
        self,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        shared_latent_dim: int = 128,
        key_latent_dim: int = 64,
        value_latent_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            sharer_num_kv_heads,
            sharer_head_dim,
            receiver_num_kv_heads,
            receiver_head_dim,
            shared_latent_dim,
            key_latent_dim,
            value_latent_dim,
            dtype,
        )
        source_width = int(sharer_num_kv_heads) * int(sharer_head_dim)
        self.source_decoder_k = nn.Sequential(
            nn.Linear(key_latent_dim, 4 * key_latent_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * key_latent_dim, source_width, dtype=dtype),
        )
        self.source_decoder_v = nn.Sequential(
            nn.Linear(value_latent_dim, 4 * value_latent_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * value_latent_dim, source_width, dtype=dtype),
        )

    def decode_source(self, key_latent: torch.Tensor, value_latent: torch.Tensor):
        if key_latent.shape != value_latent.shape or key_latent.ndim != 3:
            raise ValueError("C2C JCB latents must be matching [B,S,64] tensors.")
        batch, sequence_length, _ = key_latent.shape
        key = self.source_decoder_k(key_latent.to(self.source_decoder_k[0].weight.dtype))
        value = self.source_decoder_v(value_latent.to(self.source_decoder_v[0].weight.dtype))
        key = key.reshape(batch, sequence_length, self.sharer_num_kv_heads, self.sharer_head_dim)
        value = value.reshape(batch, sequence_length, self.sharer_num_kv_heads, self.sharer_head_dim)
        return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()
