"""Fixed, non-adaptive JCB latent DCT/int16/zlib packet codec."""

from __future__ import annotations

import time
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
        timing_mode: str = "profile",
    ) -> None:
        if beta <= 0 or low <= 0 or high < low:
            raise ValueError("Invalid JCB quantization parameters.")
        if entropy_backend != "zlib6":
            raise ValueError("The revised packet codec currently requires zlib6.")
        scale_mode = str(scale_mode).lower()
        if scale_mode not in {"none", "rms"}:
            raise ValueError("scale_mode must be 'none' or 'rms'.")
        timing_mode = str(timing_mode).lower()
        if timing_mode not in {"profile", "off"}:
            raise ValueError("timing_mode must be 'profile' or 'off'.")
        self.beta = float(beta)
        self.low = float(low)
        self.high = float(high)
        self.scale_mode = scale_mode
        self.entropy_backend = entropy_backend
        self.timing_mode = timing_mode
        self.last_encode_stats: dict[str, Any] | None = None
        self.last_decode_stats: dict[str, Any] | None = None

    def _sync(self, device: torch.device) -> None:
        if self.timing_mode == "profile" and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _timed(self, device: torch.device, fn):
        self._sync(device)
        started = time.perf_counter()
        value = fn()
        self._sync(device)
        return value, (time.perf_counter() - started) * 1000.0

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
        total_started = time.perf_counter()
        transformed, dct_ms = self._timed(
            values.device,
            lambda: _dct_ii_ortho(values.detach().to(dtype=torch.float32), axis=2),
        )
        scale = 1.0
        if self.scale_mode == "rms" and transformed.numel():
            def _scale():
                nonlocal scale
                scale = max(float(torch.sqrt(torch.mean(transformed.square())).item()), 1e-8)
                return transformed / scale
            transformed, scale_ms = self._timed(values.device, _scale)
        else:
            scale_ms = None
        table = self._table(int(values.shape[2]), transformed.device)
        symbols, quant_ms = self._timed(
            values.device,
            lambda: torch.round(transformed / (self.beta * table.view(1, 1, -1, 1))),
        )
        minimum = float(symbols.min().item()) if symbols.numel() else 0.0
        maximum = float(symbols.max().item()) if symbols.numel() else 0.0
        if minimum < -32768 or maximum > 32767:
            raise OverflowError(
                f"JCB int16 quantization overflow: range [{minimum}, {maximum}]."
            )
        def _copy_symbols():
            return symbols.to(torch.int16).cpu().contiguous().numpy()
        array, d2h_ms = self._timed(values.device, _copy_symbols)
        raw_started = time.perf_counter()
        raw = array.tobytes(order="C")
        symbol_pack_ms = (time.perf_counter() - raw_started) * 1000.0
        entropy_started = time.perf_counter()
        payload = zlib.compress(raw, level=6)
        entropy_encode_ms = (time.perf_counter() - entropy_started) * 1000.0
        self.last_encode_stats = {
            "timing_mode": self.timing_mode,
            "gpu_dct_ms": dct_ms,
            "gpu_scale_ms": scale_ms,
            "gpu_quant_ms": quant_ms,
            "d2h_ms": d2h_ms,
            "symbol_pack_ms": symbol_pack_ms,
            "entropy_encode_ms": entropy_encode_ms,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
            "raw_symbol_bytes": len(raw),
            "compressed_symbol_bytes": len(payload),
        }
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
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        total_started = time.perf_counter()
        entropy_started = time.perf_counter()
        raw = zlib.decompress(packet["data"])
        entropy_decode_ms = (time.perf_counter() - entropy_started) * 1000.0
        if len(raw) != expected * 2:
            raise ValueError("JCB packet has inconsistent int16 payload size.")
        unpack_started = time.perf_counter()
        symbols = torch.frombuffer(bytearray(raw), dtype=torch.int16).reshape(shape)
        symbol_unpack_ms = (time.perf_counter() - unpack_started) * 1000.0
        symbols, h2d_ms = self._timed(target_device, lambda: symbols.to(device=target_device, dtype=torch.float32))
        table = self._table(shape[2], symbols.device)
        transformed, dequant_ms = self._timed(target_device, lambda: symbols * (float(packet["beta"]) * table.view(1, 1, -1, 1)))
        scale_mode = str(packet.get("scale_mode", "none")).lower()
        if scale_mode not in {"none", "rms"}:
            raise ValueError(f"Unsupported JCB packet scale_mode: {scale_mode}")
        if scale_mode == "rms":
            transformed, scale_restore_ms = self._timed(target_device, lambda: transformed * float(packet.get("scale", 1.0)))
        else:
            scale_restore_ms = None
        restored, idct_ms = self._timed(target_device, lambda: _idct_iii_ortho(transformed, axis=2))
        self.last_decode_stats = {
            "timing_mode": self.timing_mode,
            "entropy_decode_ms": entropy_decode_ms,
            "symbol_unpack_ms": symbol_unpack_ms,
            "h2d_ms": h2d_ms,
            "gpu_dequant_ms": dequant_ms,
            "gpu_scale_restore_ms": scale_restore_ms,
            "gpu_idct_ms": idct_ms,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
            "raw_symbol_bytes": len(raw),
            "compressed_symbol_bytes": len(packet["data"]),
        }
        return restored
