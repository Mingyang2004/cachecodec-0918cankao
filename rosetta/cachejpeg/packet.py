"""Fixed, non-adaptive JCB latent DCT/int16/zlib packet codec."""

from __future__ import annotations

import zlib
from typing import Any

import torch


def _dct_ii_ortho(values: torch.Tensor, axis: int) -> torch.Tensor:
    moved = values.movedim(axis, -1).to(dtype=torch.float32).contiguous()
    length = int(moved.shape[-1])
    if length == 0:
        return values.to(dtype=torch.float32)
    flat = moved.reshape(-1, length)
    reordered = torch.cat((flat[:, ::2], flat[:, 1::2].flip(dims=(1,))), dim=1)
    spectrum = torch.fft.fft(reordered, dim=1)
    phase = -torch.arange(length, device=moved.device, dtype=torch.float32)
    phase = phase * torch.pi / (2.0 * length)
    coefficients = spectrum.real * torch.cos(phase) - spectrum.imag * torch.sin(phase)
    coefficients[:, :1] /= 2.0 * length**0.5
    if length > 1:
        coefficients[:, 1:] /= 2.0 * (length / 2.0) ** 0.5
    return (2.0 * coefficients).reshape(moved.shape).movedim(-1, axis)


def _idct_iii_ortho(values: torch.Tensor, axis: int) -> torch.Tensor:
    moved = values.movedim(axis, -1).to(dtype=torch.float32).contiguous()
    length = int(moved.shape[-1])
    if length == 0:
        return values.to(dtype=torch.float32)
    original_shape = moved.shape
    flat = moved.reshape(-1, length) / 2.0
    flat[:, :1] *= 2.0 * length**0.5
    if length > 1:
        flat[:, 1:] *= 2.0 * (length / 2.0) ** 0.5
    phase = torch.arange(length, device=moved.device, dtype=torch.float32)
    phase = phase * torch.pi / (2.0 * length)
    imag = torch.cat((torch.zeros_like(flat[:, :1]), -flat.flip(dims=(1,))[:, :-1]), dim=1)
    real_part = flat * torch.cos(phase) - imag * torch.sin(phase)
    imag_part = flat * torch.sin(phase) + imag * torch.cos(phase)
    reordered = torch.fft.ifft(torch.complex(real_part, imag_part), dim=1).real
    restored = torch.empty_like(reordered)
    even_count = length - length // 2
    restored[:, ::2] = reordered[:, :even_count]
    restored[:, 1::2] = reordered.flip(dims=(1,))[:, : length // 2]
    return restored.reshape(original_shape).movedim(-1, axis)


class DCTInt16PacketCodec:
    """Encode one latent pseudo-KV tensor along its sequence dimension.

    Quantization is deliberately fixed and deterministic. ``scale_mode='none'``
    uses the raw DCT coefficients; ``scale_mode='rms'`` stores one per-packet
    RMS side-info scalar and restores it after dequantization. Neither mode
    has a learned allocator, estimated rate, or rate loss.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        low: float = 1.0,
        high: float = 8.0,
        scale_mode: str = "none",
        entropy_backend: str = "zlib6",
    ) -> None:
        if beta <= 0 or low <= 0 or high < low:
            raise ValueError("Invalid JCB quantization parameters.")
        if entropy_backend != "zlib6":
            raise ValueError("The revised packet codec currently requires zlib6.")
        scale_mode = str(scale_mode).lower()
        if scale_mode not in {"none", "rms"}:
            raise ValueError("scale_mode must be 'none' or 'rms'.")
        self.beta = float(beta)
        self.low = float(low)
        self.high = float(high)
        self.scale_mode = scale_mode
        self.entropy_backend = entropy_backend

    def _table(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        if sequence_length <= 1:
            ratio = torch.zeros(sequence_length, device=device, dtype=torch.float32)
        else:
            ratio = torch.arange(sequence_length, device=device, dtype=torch.float32)
            ratio = ratio / float(sequence_length - 1)
        return self.low + (self.high - self.low) * ratio.square()

    def encode(self, values: torch.Tensor) -> dict[str, Any]:
        if values.ndim != 4:
            raise ValueError("DCTInt16PacketCodec expects [B,H,S,D] tensors.")
        transformed = _dct_ii_ortho(values.detach().to(dtype=torch.float32), axis=2)
        scale = 1.0
        if self.scale_mode == "rms" and transformed.numel():
            scale = max(float(torch.sqrt(torch.mean(transformed.square())).item()), 1e-8)
            transformed = transformed / scale
        table = self._table(int(values.shape[2]), transformed.device)
        symbols = torch.round(transformed / (self.beta * table.view(1, 1, -1, 1)))
        minimum = float(symbols.min().item()) if symbols.numel() else 0.0
        maximum = float(symbols.max().item()) if symbols.numel() else 0.0
        if minimum < -32768 or maximum > 32767:
            raise OverflowError(
                f"JCB int16 quantization overflow: range [{minimum}, {maximum}]."
            )
        array = symbols.to(torch.int16).cpu().contiguous().numpy()
        raw = array.tobytes(order="C")
        payload = zlib.compress(raw, level=6)
        return {
            "version": 1,
            "shape": tuple(int(dim) for dim in values.shape),
            "dtype": "int16",
            "beta": self.beta,
            "low": self.low,
            "high": self.high,
            "scale_mode": self.scale_mode,
            "scale": float(scale),
            "entropy_backend": self.entropy_backend,
            "data": payload,
            "raw_bytes": len(raw),
            "payload_bytes": len(payload),
            "original_bf16_bytes": int(values.numel() * 2),
        }

    def decode(self, packet: dict[str, Any], *, device: torch.device | str | None = None) -> torch.Tensor:
        if packet.get("version") != 1 or packet.get("dtype") != "int16":
            raise ValueError("Unsupported JCB int16 packet.")
        shape = tuple(int(dim) for dim in packet["shape"])
        expected = 1
        for dim in shape:
            expected *= dim
        raw = zlib.decompress(packet["data"])
        if len(raw) != expected * 2:
            raise ValueError("JCB packet has inconsistent int16 payload size.")
        symbols = torch.frombuffer(bytearray(raw), dtype=torch.int16).reshape(shape)
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        symbols = symbols.to(device=target_device, dtype=torch.float32)
        table = self._table(shape[2], symbols.device)
        transformed = symbols * (float(packet["beta"]) * table.view(1, 1, -1, 1))
        scale_mode = str(packet.get("scale_mode", "none")).lower()
        if scale_mode not in {"none", "rms"}:
            raise ValueError(f"Unsupported JCB packet scale_mode: {scale_mode}")
        if scale_mode == "rms":
            transformed = transformed * float(packet.get("scale", 1.0))
        return _idct_iii_ortho(transformed, axis=2)
