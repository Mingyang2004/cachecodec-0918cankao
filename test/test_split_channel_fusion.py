import torch

from rosetta.model.projector import create_projector
from rosetta.model.split_channel_fusion import (
    SplitChannelFusionC2CProjector,
    SplitChannelFusionProjector,
)


def _source():
    return torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)


def test_split_channel_fusion_concat_contract_and_independent_encoders():
    projector = SplitChannelFusionProjector(2, 4, 1, 4)
    key_latent, value_latent = projector.encode(_source())
    assert key_latent.shape == value_latent.shape == (1, 5, 64)
    key, value = projector.decode_transport(key_latent, value_latent)
    assert key.shape == value.shape == (1, 1, 5, 4)
    assert projector.key_encoder[0].weight.data_ptr() != projector.value_encoder[0].weight.data_ptr()
    assert not hasattr(projector, "shared_encoder")
    assert not hasattr(projector, "key_projection")
    assert not hasattr(projector, "value_projection")


def test_split_channel_fusion_c2c_restores_sharer_geometry():
    projector = SplitChannelFusionC2CProjector(2, 4, 1, 4)
    source = _source()
    key, value = projector.decode_source(*projector.encode(source))
    assert key.shape == value.shape == source[0].shape


def test_split_channel_fusion_is_registered_and_has_gradients():
    projector = create_projector(
        "SplitChannelFusionProjector",
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
    )
    key, value = projector.decode_transport(*projector.encode(_source()))
    (key.square().mean() + value.square().mean()).backward()
    assert projector.key_encoder[0].weight.grad is not None
    assert projector.value_encoder[0].weight.grad is not None
