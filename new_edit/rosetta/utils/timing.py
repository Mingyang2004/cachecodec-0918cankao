"""One timing schema for all CacheCodec evaluation routes.

All values ending in ``_ms`` are synchronized wall-clock measurements.  A
missing stage is represented by ``None`` (never by zero), so summaries do not
silently treat a non-applicable stage as a measured zero-cost stage.
"""

from __future__ import annotations

from typing import Any, Mapping


TIMING_SCHEMA_VERSION = 1


def transport_fields(stats: Any) -> dict[str, float | int | None]:
    """Convert a transport stats object to stable, JSON-ready timing fields."""
    if stats is None:
        return {
            "wire_payload_bytes": None,
            "serialize_ms": None,
            "link_model_ms": None,
            "transport_runtime_ms": None,
            "deserialize_ms": None,
            "transport_wall_ms": None,
        }
    serialize_s = float(getattr(stats, "serialize_seconds", 0.0))
    link_s = float(getattr(stats, "link_model_seconds", 0.0))
    runtime_s = float(getattr(stats, "socket_runtime_seconds", 0.0))
    deserialize_s = float(getattr(stats, "deserialize_seconds", 0.0))
    return {
        "wire_payload_bytes": int(
            getattr(stats, "frame_bytes", getattr(stats, "payload_bytes", 0))
        ),
        "serialize_ms": serialize_s * 1000.0,
        "link_model_ms": link_s * 1000.0,
        "transport_runtime_ms": runtime_s * 1000.0,
        "deserialize_ms": deserialize_s * 1000.0,
        # This is the actual local transport wall time.  Link model time is
        # already included by SocketPairTransport's sleep and must not be
        # added again when calculating a wall-clock critical path.
        "transport_wall_ms": (
            serialize_s
            + float(getattr(stats, "transmit_seconds", 0.0))
            + deserialize_s
        ) * 1000.0,
    }


def new_timing_record(
    *, route: str, bridge: str, codec: str, execution_mode: str = "serial"
) -> dict[str, Any]:
    """Return the complete per-example timing record with stable keys."""
    return {
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "route": route,
        "bridge": bridge,
        "codec": codec,
        "execution_mode": execution_mode,
        "sharer_prefill_ms": None,
        "sharer_decode_ms": None,
        "sender_projection_ms": None,
        "gpu_dct_ms": None,
        "gpu_quant_ms": None,
        "d2h_ms": None,
        "entropy_encode_ms": None,
        "serialize_ms": None,
        "link_model_ms": None,
        "transport_runtime_ms": None,
        "deserialize_ms": None,
        "wire_payload_bytes": None,
        "entropy_decode_ms": None,
        "h2d_ms": None,
        "gpu_idct_ms": None,
        "receiver_projection_or_fusion_ms": None,
        "receiver_reencode_ms": None,
        "receiver_prefill_ms": None,
        "receiver_first_token_ms": None,
        "receiver_decode_ms": None,
        "ttft_ms": None,
        "pipeline_critical_path_ms": None,
        "model_e2e_internal_ms": None,
    }


def merge_present(record: dict[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Merge only explicit values, preserving ``None`` for absent stages."""
    for key, value in values.items():
        if value is not None:
            record[key] = value
    return record
