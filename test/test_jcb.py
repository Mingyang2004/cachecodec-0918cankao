import numpy as np
import pytest
import torch

from rosetta.model.jcb import JCBProjector, JCBC2CProjector
from rosetta.cachejpeg.packet import DCTInt16PacketCodec
from rosetta.cachejpeg.fake_quant import DCTFakeQuantizer
from rosetta.model.projector import DirectConcatProjector, create_projector


def test_jcb_concat_shapes_and_independent_kv_decoders():
    projector = JCBProjector(
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
    )
    source = (torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    key_latent, value_latent = projector.encode(source)
    assert key_latent.shape == value_latent.shape == (1, 5, 64)
    key, value = projector.decode_transport(key_latent, value_latent)
    assert key.shape == value.shape == (1, 1, 5, 4)
    assert projector.decoder_k[0].weight.data_ptr() != projector.decoder_v[0].weight.data_ptr()


def test_jcb_is_registered_for_checkpoint_configs():
    projector = create_projector(
        "JCBProjector",
        sharer_num_kv_heads=1,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
    )
    assert isinstance(projector, JCBProjector)


def test_dct_int16_packet_roundtrip_and_wire_stats():
    values = torch.randn(1, 1, 9, 4)
    codec = DCTInt16PacketCodec(beta=1.0, entropy_backend="zlib6")
    packet = codec.encode(values)
    restored = codec.decode(packet)
    assert restored.shape == values.shape
    assert restored.dtype == torch.float32
    assert np.isfinite(restored.numpy()).all()
    assert packet["payload_bytes"] > 0
    assert packet["original_bf16_bytes"] == values.numel() * 2


def test_dct_int16_packet_rms_scale_is_explicit_and_roundtrips():
    values = torch.randn(1, 1, 9, 4) * 17.0
    codec = DCTInt16PacketCodec(scale_mode="rms", entropy_backend="zlib6")
    packet = codec.encode(values)
    restored = codec.decode(packet)
    assert packet["scale_mode"] == "rms"
    assert packet["scale"] > 0.0
    assert restored.shape == values.shape
    assert torch.isfinite(restored).all()


def test_dct_int16_packet_rejects_overflow_without_clipping():
    codec = DCTInt16PacketCodec(beta=1e-9, entropy_backend="zlib6")
    with pytest.raises(OverflowError):
        codec.encode(torch.full((1, 1, 4, 1), 1e9))


def test_c2c_jcb_reconstructs_sharer_geometry_without_receiver_cache():
    projector = JCBC2CProjector(
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
    )
    source = (torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    key_latent, value_latent = projector.encode(source)
    key, value = projector.decode_source(key_latent, value_latent)
    assert key.shape == value.shape == source[0].shape


def test_direct_concat_transports_full_sharer_geometry():
    projector = create_projector(
        "DirectConcatProjector",
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=8,
        hidden_dim=16,
        num_layers=2,
    )
    assert isinstance(projector, DirectConcatProjector)
    source = (torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    wire = projector.encode(source)
    key, value = projector.decode_source(*wire)
    assert key.shape == value.shape == (1, 1, 5, 8)


def test_dct_fake_quant_preserves_shape_and_uses_ste_gradient():
    quantizer = DCTFakeQuantizer(beta=1.0, low=1.0, high=8.0)
    latent = torch.randn(2, 7, 64, requires_grad=True)
    reconstructed = quantizer(latent)
    assert reconstructed.shape == latent.shape
    assert reconstructed.dtype == latent.dtype
    reconstructed.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_dct_fake_quant_matches_fixed_frequency_steps():
    quantizer = DCTFakeQuantizer(beta=1.0, low=1.0, high=8.0)
    steps = quantizer.quantization_steps(9)
    assert steps.shape == (9,)
    assert steps[0].item() == pytest.approx(1.0)
    assert steps[-1].item() == pytest.approx(8.0)
    assert torch.all(steps[1:] >= steps[:-1])
