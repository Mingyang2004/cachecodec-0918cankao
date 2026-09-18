"""Differentiable fixed DCT quantization for QAT.

This module is intentionally a training-only approximation of
``DCTInt16PacketCodec``.  It performs the same sequence-axis DCT and fixed
frequency step table, but replaces the non-differentiable round operation with
the straight-through estimator.  Serialization and zlib remain evaluation
only and therefore never enter the autograd graph.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def _dct_ii_ortho(values: Tensor) -> Tensor:
    """Apply an orthonormal DCT-II along dimension 1 of ``[B,S,C]``."""

    if values.ndim != 3:
        raise ValueError(
            "DCT fake quantization expects latent tensors with shape [B,S,C]."
        )
    batch, sequence_length, channels = values.shape
    if sequence_length == 0:
        return values.to(dtype=torch.float32)
    moved = values.transpose(1, 2).contiguous().to(dtype=torch.float32)
    flat = moved.reshape(-1, sequence_length)
    reordered = torch.cat((flat[:, ::2], flat[:, 1::2].flip(dims=(1,))), dim=1)
    spectrum = torch.fft.fft(reordered, dim=1)
    phase = -torch.arange(
        sequence_length, device=values.device, dtype=torch.float32
    )
    phase = phase * torch.pi / (2.0 * sequence_length)
    coefficients = spectrum.real * torch.cos(phase) - spectrum.imag * torch.sin(phase)
    coefficients[:, :1] /= 2.0 * sequence_length**0.5
    if sequence_length > 1:
        coefficients[:, 1:] /= 2.0 * (sequence_length / 2.0) ** 0.5
    return (2.0 * coefficients).reshape(moved.shape).transpose(1, 2).contiguous()


def _idct_iii_ortho(values: Tensor) -> Tensor:
    """Apply the matching orthonormal inverse DCT-III along dimension 1."""

    if values.ndim != 3:
        raise ValueError(
            "Inverse DCT fake quantization expects latent tensors with shape [B,S,C]."
        )
    batch, sequence_length, channels = values.shape
    if sequence_length == 0:
        return values.to(dtype=torch.float32)
    moved = values.transpose(1, 2).contiguous().to(dtype=torch.float32)
    original_shape = moved.shape
    flat = moved.reshape(-1, sequence_length) / 2.0
    flat[:, :1] *= 2.0 * sequence_length**0.5
    if sequence_length > 1:
        flat[:, 1:] *= 2.0 * (sequence_length / 2.0) ** 0.5
    phase = torch.arange(sequence_length, device=values.device, dtype=torch.float32)
    phase = phase * torch.pi / (2.0 * sequence_length)
    imag = torch.cat((torch.zeros_like(flat[:, :1]), -flat.flip(dims=(1,))[:, :-1]), dim=1)
    real_part = flat * torch.cos(phase) - imag * torch.sin(phase)
    imag_part = flat * torch.sin(phase) + imag * torch.cos(phase)
    reordered = torch.fft.ifft(torch.complex(real_part, imag_part), dim=1).real
    restored = torch.empty_like(reordered)
    even_count = sequence_length - sequence_length // 2
    restored[:, ::2] = reordered[:, :even_count]
    restored[:, 1::2] = reordered.flip(dims=(1,))[:, : sequence_length // 2]
    return restored.reshape(original_shape).transpose(1, 2).contiguous()


class DCTFakeQuantizer(nn.Module):
    """Fixed sequence-DCT quantizer used between JCB encode and decode.

    The quantization table is ``beta * (low + (high-low) * r**2)`` where
    ``r`` runs from zero to one over sequence frequency.  ``forward`` returns
    IDCT(requantized DCT(latent)) with an STE, preserving the gradient of the
    unquantized latent while exposing quantization error to the forward pass.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        low: float = 1.0,
        high: float = 8.0,
        straight_through: bool = True,
    ) -> None:
        super().__init__()
        if beta <= 0 or low <= 0 or high < low:
            raise ValueError("Invalid DCT fake-quantization parameters.")
        self.beta = float(beta)
        self.low = float(low)
        self.high = float(high)
        self.straight_through = bool(straight_through)
        self.last_stats: dict[str, Any] | None = None

    def quantization_steps(self, sequence_length: int, device: torch.device | None = None) -> Tensor:
        if sequence_length < 0:
            raise ValueError("sequence_length must be non-negative.")
        if sequence_length <= 1:
            ratio = torch.zeros(sequence_length, device=device, dtype=torch.float32)
        else:
            ratio = torch.arange(sequence_length, device=device, dtype=torch.float32)
            ratio = ratio / float(sequence_length - 1)
        return self.beta * (self.low + (self.high - self.low) * ratio.square())

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 3:
            raise ValueError(
                "DCTFakeQuantizer expects latent tensors with shape [B,S,C]."
            )
        original_dtype = latent.dtype
        transformed = _dct_ii_ortho(latent)
        steps = self.quantization_steps(
            int(latent.shape[1]), device=transformed.device
        ).view(1, -1, 1)
        quantized = torch.round(transformed / steps) * steps
        reconstructed = _idct_iii_ortho(quantized)
        if self.straight_through:
            output = latent.to(dtype=torch.float32) + (
                reconstructed - latent.to(dtype=torch.float32)
            ).detach()
        else:
            output = reconstructed
        self.last_stats = {
            "sequence_length": int(latent.shape[1]),
            "channels": int(latent.shape[2]),
            "quantization_mse": float(
                (reconstructed - latent.detach().to(dtype=torch.float32))
                .pow(2)
                .mean()
                .item()
            )
            if latent.numel()
            else 0.0,
            "beta": self.beta,
            "low": self.low,
            "high": self.high,
            "straight_through": self.straight_through,
        }
        return output.to(dtype=original_dtype)


__all__ = ["DCTFakeQuantizer"]
