"""Independent K/V latent transport projectors for split-channel fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rosetta.model.projector import Projector
from rosetta.utils.registry import capture_init_args, register_model


@register_model
@capture_init_args
class SplitChannelFusionProjector(Projector):
    """Independent ``Hs*Ds -> 64 -> 256 -> 64`` K/V encoders."""

    def __init__(
        self, sharer_num_kv_heads: int, sharer_head_dim: int,
        receiver_num_kv_heads: int, receiver_head_dim: int,
        key_latent_dim: int = 64, value_latent_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if key_latent_dim != 64 or value_latent_dim != 64:
            raise ValueError("SplitChannelFusion uses fixed 64-wide K/V latents.")
        self.sharer_num_kv_heads = int(sharer_num_kv_heads)
        self.sharer_head_dim = int(sharer_head_dim)
        self.receiver_num_kv_heads = int(receiver_num_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.key_latent_dim = int(key_latent_dim)
        self.value_latent_dim = int(value_latent_dim)
        source_channels = self.sharer_num_kv_heads * self.sharer_head_dim
        receiver_channels = self.receiver_num_kv_heads * self.receiver_head_dim

        def encoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(source_channels, 64, dtype=dtype), nn.GELU(),
                nn.Linear(64, 256, dtype=dtype), nn.GELU(),
                nn.Linear(256, 64, dtype=dtype),
            )

        def receiver_decoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(64, 256, dtype=dtype), nn.GELU(),
                nn.Linear(256, receiver_channels, dtype=dtype),
            )

        self.key_encoder = encoder()
        self.value_encoder = encoder()
        self.decoder_k = receiver_decoder()
        self.decoder_v = receiver_decoder()

    def _encode_tensor(self, tensor: Tensor, encoder: nn.Module) -> Tensor:
        if tensor.ndim != 4:
            raise ValueError("SplitChannelFusion expects [B,H,S,D] tensors.")
        batch, heads, sequence_length, head_dim = tensor.shape
        if (heads, head_dim) != (self.sharer_num_kv_heads, self.sharer_head_dim):
            raise ValueError("Sharer KV geometry does not match SplitChannelFusion.")
        channels = tensor.transpose(1, 2).contiguous().reshape(batch, sequence_length, -1)
        return encoder(channels.to(dtype=next(encoder.parameters()).dtype))

    def encode(self, source_kv: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        key, value = source_kv
        if key.shape != value.shape:
            raise ValueError("SplitChannelFusion Sharer K/V shapes must match.")
        return self._encode_tensor(key, self.key_encoder), self._encode_tensor(value, self.value_encoder)

    def _decode_tensor(self, latent: Tensor, decoder: nn.Module, *, num_heads: int, head_dim: int) -> Tensor:
        if latent.ndim != 3 or latent.shape[-1] != 64:
            raise ValueError("SplitChannelFusion latents must have shape [B,S,64].")
        batch, sequence_length, _ = latent.shape
        channels = decoder(latent.to(dtype=next(decoder.parameters()).dtype))
        return channels.reshape(batch, sequence_length, num_heads, head_dim).transpose(1, 2).contiguous()

    def decode_transport(self, key_latent: Tensor, value_latent: Tensor) -> tuple[Tensor, Tensor]:
        if key_latent.shape[:2] != value_latent.shape[:2]:
            raise ValueError("SplitChannelFusion K/V latents must share batch and sequence dimensions.")
        return (
            self._decode_tensor(key_latent, self.decoder_k, num_heads=self.receiver_num_kv_heads, head_dim=self.receiver_head_dim),
            self._decode_tensor(value_latent, self.decoder_v, num_heads=self.receiver_num_kv_heads, head_dim=self.receiver_head_dim),
        )

    def forward(self, source_kv: tuple[Tensor, Tensor], target_kv: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        del target_kv
        return self.decode_transport(*self.encode(source_kv))


@register_model
@capture_init_args
class SplitChannelFusionC2CProjector(SplitChannelFusionProjector):
    """Split-channel codec that restores Sharer geometry for the C2C fuser."""

    def __init__(
        self, sharer_num_kv_heads: int, sharer_head_dim: int,
        receiver_num_kv_heads: int, receiver_head_dim: int,
        key_latent_dim: int = 64, value_latent_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            sharer_num_kv_heads=sharer_num_kv_heads,
            sharer_head_dim=sharer_head_dim,
            receiver_num_kv_heads=receiver_num_kv_heads,
            receiver_head_dim=receiver_head_dim,
            key_latent_dim=key_latent_dim,
            value_latent_dim=value_latent_dim,
            dtype=dtype,
        )
        source_channels = self.sharer_num_kv_heads * self.sharer_head_dim

        def source_decoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(64, 256, dtype=dtype), nn.GELU(),
                nn.Linear(256, source_channels, dtype=dtype),
            )

        self.source_decoder_k = source_decoder()
        self.source_decoder_v = source_decoder()

    def decode_source(self, key_latent: Tensor, value_latent: Tensor) -> tuple[Tensor, Tensor]:
        if key_latent.shape != value_latent.shape:
            raise ValueError("SplitChannelFusion C2C K/V latents must have matching shapes.")
        return (
            self._decode_tensor(key_latent, self.source_decoder_k, num_heads=self.sharer_num_kv_heads, head_dim=self.sharer_head_dim),
            self._decode_tensor(value_latent, self.source_decoder_v, num_heads=self.sharer_num_kv_heads, head_dim=self.sharer_head_dim),
        )
