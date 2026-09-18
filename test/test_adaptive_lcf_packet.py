import torch
import struct

from rosetta.cachejpeg_rosetta.adaptive_lcf_packet import AdaptiveLCFPacketCodec
from rosetta.model.adaptive_quant_table import (
    ProjectedKVAdaptiveQuantizer,
    resolve_adaptive_quant_table_config,
)


def _cache():
    return tuple(
        (
            torch.randn(1, 1, 7, 4),
            torch.randn(1, 1, 7, 4),
        )
        for _ in range(2)
    )


def test_adaptive_packet_roundtrip_matches_eval_quantizer():
    config = resolve_adaptive_quant_table_config(
        {
            "enabled": True,
            "allocator_type": "projected_layer_kv",
            "feature_bands": 2,
            "hidden_dim": 16,
            "scale_side_info_bits": 32,
        }
    )
    quantizer = ProjectedKVAdaptiveQuantizer(num_layers=2, config=config).eval()
    source = _cache()
    expected = quantizer(source)
    codec = AdaptiveLCFPacketCodec(quantizer)

    packet = codec.encode(source)
    decoded = codec.decode(packet, device=torch.device("cpu"))

    assert packet.startswith(b"ALCF")
    assert decoded.table_indices.equal(expected.table_indices)
    torch.testing.assert_close(decoded.scale, expected.scale)
    torch.testing.assert_close(decoded.rounded_symbols, expected.rounded_symbols)
    for actual, expected_layer in zip(decoded.past_key_values, expected.past_key_values):
        torch.testing.assert_close(actual[0], expected_layer[0])
        torch.testing.assert_close(actual[1], expected_layer[1])


def test_adaptive_packet_rejects_wrong_quantizer_geometry():
    config = resolve_adaptive_quant_table_config(
        {"enabled": True, "allocator_type": "projected_layer_kv"}
    )
    quantizer = ProjectedKVAdaptiveQuantizer(num_layers=2, config=config).eval()
    codec = AdaptiveLCFPacketCodec(quantizer)
    with torch.no_grad():
        packet = codec.encode(_cache())
    broken = packet[:-1]
    try:
        codec.decode(broken, device=torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("truncated adaptive packet must be rejected")


def test_adaptive_packet_rejects_unsupported_wire_dtypes():
    config = resolve_adaptive_quant_table_config(
        {"enabled": True, "allocator_type": "projected_layer_kv"}
    )
    codec = AdaptiveLCFPacketCodec(
        ProjectedKVAdaptiveQuantizer(num_layers=2, config=config).eval()
    )
    packet = codec.encode(_cache())
    header_size = struct.calcsize("<4sBI")
    metadata_length = struct.unpack("<I", packet[5:header_size])[0]
    metadata_start = header_size
    metadata_end = metadata_start + metadata_length
    metadata = packet[metadata_start:metadata_end]
    broken_metadata = metadata.replace(b'"scale_dtype":"float32_le"', b'"scale_dtype":"float16_le"')
    assert len(broken_metadata) == len(metadata)
    broken = packet[:metadata_start] + broken_metadata + packet[metadata_end:]
    try:
        codec.decode(broken, device=torch.device("cpu"))
    except ValueError as error:
        assert "dtype" in str(error)
    else:
        raise AssertionError("unsupported packet dtype must be rejected")
