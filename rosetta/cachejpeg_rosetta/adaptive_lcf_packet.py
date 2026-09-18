"""Lossless packet transport for the projected-KV adaptive QAT path."""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from rosetta.model.adaptive_quant_table import (
    AdaptiveQuantTableResult,
    ProjectedKVAdaptiveQuantizer,
    _idct_iii_ortho,
)


@dataclass(frozen=True)
class AdaptiveLCFPacketDecode:
    past_key_values: tuple[tuple[Tensor, Tensor], ...]
    table_indices: Tensor
    scale: Tensor
    rounded_symbols: Tensor
    metadata: dict[str, Any]


class AdaptiveLCFPacketCodec:
    """Serialize projected transport latents after adaptive quantization.

    Quantization remains lossy; alpha IDs, scales, and integer DCT symbols are
    transported losslessly with a small JSON header and zlib symbol stream.
    """

    _MAGIC = b"ALCF"
    _VERSION = 1
    _HEADER = struct.Struct("<4sBI")

    def __init__(self, quantizer: ProjectedKVAdaptiveQuantizer):
        if not isinstance(quantizer, ProjectedKVAdaptiveQuantizer):
            raise TypeError("Adaptive LCF packet codec requires ProjectedKVAdaptiveQuantizer.")
        self.quantizer = quantizer

    @property
    def num_layers(self) -> int:
        return self.quantizer.num_layers

    def _metadata(self, result: AdaptiveQuantTableResult) -> dict[str, Any]:
        symbols = result.rounded_symbols
        return {
            "version": self._VERSION,
            "batch_size": int(symbols.shape[0]),
            "num_layers": int(symbols.shape[1]),
            "kv_types": int(symbols.shape[2]),
            "pseudo_heads": int(symbols.shape[3]),
            "sequence_length": int(symbols.shape[4]),
            "latent_dim": int(symbols.shape[5]),
            "alpha_candidates": [
                float(value) for value in self.quantizer.alpha_candidates.detach().cpu().tolist()
            ],
            "q_base_min": float(self.quantizer.config.q_base_min),
            "q_base_max": float(self.quantizer.config.q_base_max),
            "q_base_power": float(self.quantizer.config.q_base_power),
            "scale_dtype": "float32_le",
            "symbol_dtype": "int32_le",
            "entropy_coder": "zlib",
            "zlib_level": 6,
        }

    @torch.no_grad()
    def encode(
        self, past_key_values: Sequence[tuple[Tensor, Tensor]]
    ) -> bytes:
        self.quantizer.eval()
        result = self.quantizer(past_key_values)
        metadata = self._metadata(result)
        table_indices = result.table_indices.detach().cpu().to(torch.uint8).contiguous()
        scale = result.scale.detach().cpu().to(torch.float32).contiguous()
        symbols = result.rounded_symbols.detach().cpu().to(torch.int32).contiguous()
        int32 = torch.iinfo(torch.int32)
        if bool((result.rounded_symbols < int32.min).any()) or bool(
            (result.rounded_symbols > int32.max).any()
        ):
            raise OverflowError("Adaptive LCF symbols exceed int32 packet range.")
        alpha_bytes = table_indices.numpy().tobytes(order="C")
        scale_bytes = scale.numpy().astype("<f4", copy=False).tobytes(order="C")
        symbol_bytes = symbols.numpy().astype("<i4", copy=False).tobytes(order="C")
        compressed_symbols = zlib.compress(symbol_bytes, level=6)
        metadata.update(
            {
                "alpha_bytes": len(alpha_bytes),
                "scale_bytes": len(scale_bytes),
                "raw_symbol_bytes": len(symbol_bytes),
            }
        )
        metadata_bytes = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return (
            self._HEADER.pack(self._MAGIC, self._VERSION, len(metadata_bytes))
            + metadata_bytes
            + alpha_bytes
            + scale_bytes
            + compressed_symbols
        )

    def _read(self, packet: bytes, device: torch.device):
        if len(packet) < self._HEADER.size:
            raise ValueError("Adaptive LCF packet is truncated before its header.")
        magic, version, metadata_length = self._HEADER.unpack(packet[: self._HEADER.size])
        if magic != self._MAGIC or version != self._VERSION:
            raise ValueError("Unsupported adaptive LCF packet header.")
        metadata_start = self._HEADER.size
        metadata_end = metadata_start + int(metadata_length)
        if metadata_end > len(packet):
            raise ValueError("Adaptive LCF packet metadata is truncated.")
        try:
            metadata = json.loads(packet[metadata_start:metadata_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Adaptive LCF packet metadata is invalid.") from error
        if not isinstance(metadata, dict):
            raise ValueError("Adaptive LCF packet metadata must be an object.")
        if metadata.get("version") != self._VERSION:
            raise ValueError("Adaptive LCF packet metadata version is unsupported.")
        if metadata.get("scale_dtype") != "float32_le":
            raise ValueError("Adaptive LCF packet scale dtype is unsupported.")
        if metadata.get("symbol_dtype") != "int32_le":
            raise ValueError("Adaptive LCF packet symbol dtype is unsupported.")
        if metadata.get("entropy_coder") != "zlib":
            raise ValueError("Adaptive LCF packet entropy coder is unsupported.")
        try:
            expected_shape = tuple(
                int(metadata[name])
                for name in (
                    "batch_size",
                    "num_layers",
                    "kv_types",
                    "pseudo_heads",
                    "sequence_length",
                    "latent_dim",
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Adaptive LCF packet geometry metadata is invalid.") from error
        if any(size <= 0 for size in expected_shape):
            raise ValueError("Adaptive LCF packet geometry must be positive.")
        if expected_shape[1:4] != (self.num_layers, 2, 1):
            raise ValueError(f"Adaptive LCF packet geometry is incompatible: {expected_shape}.")
        expected_alpha_bytes = expected_shape[0] * expected_shape[1] * expected_shape[2] * expected_shape[3]
        expected_scale_bytes = expected_alpha_bytes * 4
        expected_symbol_bytes = 4
        for size in expected_shape:
            expected_symbol_bytes *= size
        try:
            alpha_bytes = int(metadata["alpha_bytes"])
            scale_bytes = int(metadata["scale_bytes"])
            raw_symbol_bytes = int(metadata["raw_symbol_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Adaptive LCF packet field-size metadata is invalid.") from error
        alpha_start = metadata_end
        alpha_end = alpha_start + alpha_bytes
        scale_end = alpha_end + scale_bytes
        if (alpha_bytes != expected_alpha_bytes or
                scale_bytes != expected_scale_bytes or
                raw_symbol_bytes != expected_symbol_bytes or
                scale_end > len(packet)):
            raise ValueError("Adaptive LCF packet field sizes are inconsistent.")
        try:
            decompressor = zlib.decompressobj()
            symbol_bytes = decompressor.decompress(packet[scale_end:])
            symbol_bytes += decompressor.flush()
        except zlib.error as error:
            raise ValueError("Adaptive LCF symbol stream is invalid.") from error
        if not decompressor.eof:
            raise ValueError("Adaptive LCF symbol stream is truncated.")
        if decompressor.unused_data or decompressor.unconsumed_tail:
            raise ValueError("Adaptive LCF symbol stream contains trailing data.")
        if len(symbol_bytes) != expected_symbol_bytes:
            raise ValueError("Adaptive LCF symbol stream has the wrong size.")
        table_indices = torch.frombuffer(
            bytearray(packet[alpha_start:alpha_end]), dtype=torch.uint8
        ).clone().reshape(expected_shape[:4]).to(device)
        scale = torch.frombuffer(
            bytearray(packet[alpha_end:scale_end]), dtype=torch.float32
        ).clone().reshape(expected_shape[:4]).to(device)
        symbols = torch.frombuffer(
            bytearray(symbol_bytes), dtype=torch.int32
        ).clone().reshape(expected_shape).to(device)
        return metadata, table_indices, scale, symbols

    @torch.no_grad()
    def decode(self, packet: bytes, *, device: torch.device) -> AdaptiveLCFPacketDecode:
        metadata, table_indices, scale, symbols = self._read(packet, device)
        candidates = torch.tensor(
            metadata["alpha_candidates"], device=device, dtype=torch.float32
        )
        configured = self.quantizer.alpha_candidates.to(device=device, dtype=torch.float32)
        if not torch.equal(candidates, configured):
            raise ValueError("Adaptive LCF packet alpha candidates do not match the checkpoint.")
        if bool((table_indices >= candidates.numel()).any()):
            raise ValueError("Adaptive LCF packet contains an invalid alpha table index.")
        expected_q_base = (
            float(self.quantizer.config.q_base_min),
            float(self.quantizer.config.q_base_max),
            float(self.quantizer.config.q_base_power),
        )
        packet_q_base = (
            float(metadata["q_base_min"]),
            float(metadata["q_base_max"]),
            float(metadata["q_base_power"]),
        )
        if packet_q_base != expected_q_base:
            raise ValueError("Adaptive LCF packet q_base settings do not match the checkpoint.")
        alpha = candidates[table_indices.long()]
        frequency = torch.linspace(
            0.0,
            1.0,
            symbols.shape[-2],
            device=device,
            dtype=torch.float32,
        )
        q_base = self.quantizer.config.q_base_min + (
            self.quantizer.config.q_base_max - self.quantizer.config.q_base_min
        ) * frequency.pow(self.quantizer.config.q_base_power)
        coefficients = symbols.float() * alpha.unsqueeze(-1).unsqueeze(-1)
        coefficients = coefficients * q_base.view(1, 1, 1, 1, -1, 1)
        coefficients = coefficients * scale.float().unsqueeze(-1).unsqueeze(-1)
        reconstructed = _idct_iii_ortho(coefficients, axis=4)
        cache = reconstructed.reshape(
            symbols.shape[0],
            symbols.shape[1],
            symbols.shape[2],
            symbols.shape[3],
            symbols.shape[4],
            symbols.shape[5],
        )
        past_key_values = tuple(
            (cache[:, layer, 0], cache[:, layer, 1])
            for layer in range(self.num_layers)
        )
        return AdaptiveLCFPacketDecode(
            past_key_values=past_key_values,
            table_indices=table_indices,
            scale=scale,
            rounded_symbols=symbols.float(),
            metadata=metadata,
        )
