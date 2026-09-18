import pytest
import torch

from rosetta.cachejpeg_rosetta.fuser_bridge import LoadedRosettaAssets
from rosetta.cachejpeg_rosetta.projected_kv_cache_aligner import (
    ProjectedKVConcatCacheAligner,
)
from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    ProjectedKVAdaptiveQuantizer,
    build_adaptive_quantizer,
    resolve_adaptive_quant_table_config,
)
from rosetta.model.lcf_projected_kv import LCFProjectedKVProjector
from rosetta.model.projector import create_projector


class DummyConfig:
    num_hidden_layers = 2


class DummyModel(torch.nn.Module):
    config = DummyConfig()

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _projector():
    return LCFProjectedKVProjector(
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
        shared_latent_dim=8,
        key_latent_dim=4,
        value_latent_dim=4,
    )


def _source_cache():
    return tuple(
        (
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
        )
        for _ in range(2)
    )


def _projected_transport_cache(projectors=None):
    projectors = projectors or [_projector(), _projector()]
    source = _source_cache()
    return tuple(
        tuple(latent.unsqueeze(1) for latent in projector.encode(source[layer_index]))
        for layer_index, projector in enumerate(projectors)
    )


def test_projected_kv_uses_full_shared_latent_and_has_expected_shapes():
    projector = _projector()
    source_key, source_value = _source_cache()[0]

    shared = projector.encode_shared((source_key, source_value))
    key_latent, value_latent = projector.project_transport(shared)
    receiver_key, receiver_value = projector.decode_transport(key_latent, value_latent)

    assert shared.shape == (1, 3, 8)
    assert key_latent.shape == (1, 3, 4)
    assert value_latent.shape == (1, 3, 4)
    assert receiver_key.shape == receiver_value.shape == (1, 1, 3, 4)
    assert projector.key_projection.in_features == 8
    assert projector.value_projection.in_features == 8
    assert projector.key_projection.weight.data_ptr() != projector.value_projection.weight.data_ptr()


def test_projected_kv_is_available_through_projector_registry():
    projector = create_projector(
        "LCFProjectedKVProjector",
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
        shared_latent_dim=8,
        key_latent_dim=4,
        value_latent_dim=4,
    )
    assert isinstance(projector, LCFProjectedKVProjector)


def test_projected_kv_aligner_exposes_one_head_pseudo_cache(monkeypatch):
    projectors = [_projector(), _projector()]
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=projectors,
        projector_dict={0: {1: {0: [(1, 0)], 1: [(0, 1)]}}},
    )
    monkeypatch.setattr(
        "rosetta.cachejpeg_rosetta.projected_kv_cache_aligner.apply_receiver_compact_rope",
        lambda _model, cache: cache,
    )
    aligner = ProjectedKVConcatCacheAligner(assets)

    pseudo_cache, routing = aligner.encode(_source_cache())
    prefix = aligner.decode(pseudo_cache, routing)

    assert pseudo_cache[0][0].shape == pseudo_cache[0][1].shape == (1, 1, 3, 4)
    assert prefix.key_cache[0].shape == (1, 1, 3, 4)
    assert aligner.last_alignment_stats["concat_projector_type"] == "lcf_projected_kv"


def test_projected_kv_aligner_streaming_layer_api_matches_routes(monkeypatch):
    projectors = [_projector(), _projector()]
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=projectors,
        projector_dict={0: {1: {0: [(1, 0)], 1: [(0, 1)]}}},
    )
    monkeypatch.setattr(
        "rosetta.cachejpeg_rosetta.projected_kv_cache_aligner.apply_receiver_compact_rope",
        lambda _model, cache: cache,
    )
    aligner = ProjectedKVConcatCacheAligner(assets)
    routing = aligner.prepare_routing()
    source = _source_cache()
    decoded = {}
    for route in routing.routes:
        target_layer, source_layer, _projector_idx = route
        key_latent, value_latent = aligner.encode_layer(route, *source[source_layer])
        decoded[target_layer] = aligner.decode_layer(
            route, key_latent, value_latent
        )
    prefix = aligner.assemble_receiver_cache(decoded, routing)

    assert [route[1] for route in routing.routes] == [1, 0]
    assert prefix.key_cache[0].shape == (1, 1, 3, 4)
    assert aligner.last_alignment_stats["transport_dim"] == 4


def test_projected_kv_qat_preserves_gradients_through_both_projections():
    projector = _projector()
    source_key, source_value = _source_cache()[0]
    key_latent, value_latent = projector.encode((source_key, source_value))
    quantizer = AdaptiveCoefficientQuantizer(
        num_layers=1,
        num_kv_heads=1,
        config=resolve_adaptive_quant_table_config(
            {"enabled": True, "feature_bands": 2, "hidden_dim": 8}
        ),
    )
    quantized = quantizer(((key_latent.unsqueeze(1), value_latent.unsqueeze(1)),))
    receiver_key, receiver_value = projector.decode_transport(
        quantized.past_key_values[0][0].squeeze(1),
        quantized.past_key_values[0][1].squeeze(1),
    )

    (receiver_key.square().mean() + receiver_value.square().mean()).backward()

    assert projector.key_projection.weight.grad is not None
    assert projector.value_projection.weight.grad is not None
    assert any(parameter.grad is not None for parameter in projector.shared_encoder.parameters())


def test_projected_allocator_uses_one_group_per_layer_and_kv_type():
    quantizer = ProjectedKVAdaptiveQuantizer(
        num_layers=2,
        config=resolve_adaptive_quant_table_config(
            {
                "enabled": True,
                "allocator_type": "projected_layer_kv",
                "feature_bands": 8,
                "hidden_dim": 128,
            }
        ),
    ).eval()

    result = quantizer(_projected_transport_cache())

    assert quantizer.num_groups == 4
    assert quantizer.local_encoder[0].in_features == 52
    assert result.alpha.shape == (1, 2, 2, 1)
    assert result.table_indices.shape == (1, 2, 2, 1)
    assert result.rounded_symbols.shape == (1, 2, 2, 1, 3, 4)
    assert result.estimated_payload_bits.item() > 0


def test_projected_allocator_eval_is_deterministic_and_rejects_real_multihead_cache():
    quantizer = ProjectedKVAdaptiveQuantizer(
        num_layers=2,
        config=resolve_adaptive_quant_table_config(
            {"enabled": True, "allocator_type": "projected_layer_kv"}
        ),
    ).eval()
    cache = _projected_transport_cache()

    first = quantizer(cache)
    second = quantizer(cache)

    assert torch.equal(first.table_indices, second.table_indices)
    for first_layer, second_layer in zip(first.past_key_values, second.past_key_values):
        torch.testing.assert_close(first_layer[0], second_layer[0])
        torch.testing.assert_close(first_layer[1], second_layer[1])

    multihead_cache = tuple(
        (key.expand(-1, 2, -1, -1), value.expand(-1, 2, -1, -1))
        for key, value in cache
    )
    with pytest.raises(ValueError, match="one pseudo head"):
        quantizer(multihead_cache)


def test_projected_allocator_task_and_rate_gradients_reach_all_learned_modules():
    torch.manual_seed(5)
    projectors = [_projector(), _projector()]
    quantizer = ProjectedKVAdaptiveQuantizer(
        num_layers=2,
        config=resolve_adaptive_quant_table_config(
            {
                "enabled": True,
                "allocator_type": "projected_layer_kv",
                "feature_bands": 2,
                "hidden_dim": 16,
                "rate_weight": 1e-6,
            }
        ),
    ).train()
    quantized = quantizer(_projected_transport_cache(projectors))
    receiver_cache = tuple(
        projector.decode_transport(
            quantized.past_key_values[layer_index][0].squeeze(1),
            quantized.past_key_values[layer_index][1].squeeze(1),
        )
        for layer_index, projector in enumerate(projectors)
    )
    task_loss = sum(
        key.square().mean() + value.square().mean()
        for key, value in receiver_cache
    )

    (task_loss + quantizer.rate_weight(100) * quantized.estimated_payload_bits).backward()

    for projector in projectors:
        assert projector.key_projection.weight.grad is not None
        assert projector.value_projection.weight.grad is not None
        assert any(
            parameter.grad is not None
            for parameter in projector.shared_encoder.parameters()
        )
    assert quantizer.alpha_head[-1].weight.grad is not None
    assert quantizer.log_entropy_scales.grad is not None


def test_quantizer_factory_selects_projected_allocator_without_breaking_legacy_mode():
    projected_config = resolve_adaptive_quant_table_config(
        {"enabled": True, "allocator_type": "projected_layer_kv"}
    )
    projected = build_adaptive_quantizer(
        num_layers=2,
        num_kv_heads=1,
        config=projected_config,
        cache_alignment="concat",
        concat_projector_type="lcf_projected_kv",
    )
    legacy = build_adaptive_quantizer(
        num_layers=2,
        num_kv_heads=2,
        config=resolve_adaptive_quant_table_config({"enabled": True}),
        cache_alignment="fuser",
        concat_projector_type=None,
    )

    assert isinstance(projected, ProjectedKVAdaptiveQuantizer)
    assert isinstance(legacy, AdaptiveCoefficientQuantizer)

    with pytest.raises(ValueError, match="requires concat"):
        build_adaptive_quantizer(
            num_layers=2,
            num_kv_heads=1,
            config=projected_config,
            cache_alignment="fuser",
            concat_projector_type=None,
        )
