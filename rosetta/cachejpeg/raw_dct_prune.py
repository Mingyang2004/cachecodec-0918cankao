"""Float DCT low-pass packets for the raw-KV truncation ablation.

This module deliberately has no quantizer, scale factor, entropy coder, JCB,
or anchor handling.  It serializes only each K/V tensor's low-frequency DCT
prefix; the receiver zero-fills the omitted high frequencies and applies IDCT.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from rosetta.cachejpeg.packet import _dct_ii_ortho, _idct_iii_ortho


class RawDCTPrunePacketCodec:
    def __init__(self, *, prune_ratio: float) -> None:
        if not 0.0 <= float(prune_ratio) <= 1.0:
            raise ValueError("raw_dct_prune.prune_ratio must be in [0, 1].")
        self.prune_ratio = float(prune_ratio)
        self.last_stats: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "RawDCTPrunePacketCodec | None":
        config = dict(config or {})
        if not bool(config.get("enabled", False)):
            return None
        return cls(prune_ratio=float(config.get("prune_ratio", 0.0)))

    def _encode_tensor(self, values: torch.Tensor) -> dict[str, Any]:
        if values.ndim != 4:
            raise ValueError("Raw DCT pruning expects KV tensors [B,H,S,D].")
        sequence_length = int(values.shape[2])
        if self.prune_ratio == 0.0:
            return {
                "identity": True,
                "shape": tuple(int(dim) for dim in values.shape),
                "values": values.detach().cpu().contiguous(),
            }
        # At exactly 100% pruning, transmit no DC coefficient either.  For any
        # smaller ratio retain at least one low-frequency coefficient.
        keep_length = (
            0 if self.prune_ratio == 1.0
            else max(1, int(math.ceil(sequence_length * (1.0 - self.prune_ratio))))
        )
        coefficients = _dct_ii_ortho(values.detach(), axis=2)
        return {
            "identity": False,
            "shape": tuple(int(dim) for dim in values.shape),
            "coefficients_low": coefficients[:, :, :keep_length, :].cpu().contiguous(),
            "original_freq_len": sequence_length,
            "stored_freq_len": keep_length,
        }

    @staticmethod
    def _decode_tensor(packet: dict[str, Any], reference: torch.Tensor) -> torch.Tensor:
        if bool(packet.get("identity", False)):
            return packet["values"].to(device=reference.device, dtype=reference.dtype)
        shape = tuple(int(dim) for dim in packet["shape"])
        if shape != tuple(reference.shape):
            raise ValueError("Raw DCT packet shape does not match receiver-side cache metadata.")
        coefficients = torch.zeros(shape, device=reference.device, dtype=torch.float32)
        stored_length = int(packet["stored_freq_len"])
        coefficients[:, :, :stored_length, :] = packet["coefficients_low"].to(
            device=reference.device, dtype=torch.float32
        )
        return _idct_iii_ortho(coefficients, axis=2).to(dtype=reference.dtype)

    def encode(self, cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> dict[str, Any]:
        layers = []
        for key, value in cache:
            layers.append({"key": self._encode_tensor(key), "value": self._encode_tensor(value)})
        retained = [int(layer["key"].get("stored_freq_len", layer["key"]["shape"][2])) for layer in layers]
        original = [int(layer["key"]["shape"][2]) for layer in layers]
        self.last_stats = {
            "enabled": True,
            "prune_ratio": self.prune_ratio,
            "prune_percentage": self.prune_ratio * 100.0,
            "original_freq_lengths": original,
            "stored_freq_lengths": retained,
            "retained_ratio": 1.0 - self.prune_ratio,
            "transform": "float_dct_zero_fill_idct",
            "quantization": "none",
        }
        return {"method": "raw_dct_prune_float", "version": 1, "layers": layers}

    def decode(self, packet: dict[str, Any], reference_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]):
        if packet.get("method") != "raw_dct_prune_float" or packet.get("version") != 1:
            raise ValueError("Unsupported raw DCT prune packet.")
        if len(packet["layers"]) != len(reference_cache):
            raise ValueError("Raw DCT packet has a different number of cache layers.")
        return tuple(
            (self._decode_tensor(layer["key"], ref_key), self._decode_tensor(layer["value"], ref_value))
            for layer, (ref_key, ref_value) in zip(packet["layers"], reference_cache)
        )
