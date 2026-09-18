"""Receiver-prefix alignment for historical DirectPreRopeMLPProjector checkpoints."""
from __future__ import annotations

from transformers.cache_utils import DynamicCache

from rosetta.model.direct_pre_rope_mlp import DirectPreRopeMLPProjector
from .cache_aligner import ConcatCacheAligner
from .pre_rope import apply_receiver_compact_rope


class DirectMLPConcatCacheAligner:
    def __init__(self, assets):
        self.assets = assets
        self.projector_dict = ConcatCacheAligner._convert_dict_keys_to_ints(assets.projector_dict)
        self.last_alignment_stats = None

    def align(self, sharer_cache):
        sharer_cache = ConcatCacheAligner._to_dynamic_cache(sharer_cache)
        routes = []
        mapping = self.projector_dict[int(self.assets.base_model_idx)][int(self.assets.teacher_model_idx)]
        for target, entry in mapping.items():
            source, projector_index = ConcatCacheAligner._normalize_pair((entry if isinstance(entry, list) else [entry])[0])
            routes.append((int(target), int(source), int(projector_index)))
        expected = set(range(int(self.assets.base_model.config.num_hidden_layers)))
        if {target for target, _, _ in routes} != expected:
            raise ValueError("Direct MLP concat is missing receiver-layer routes.")
        projected, prefix_length = {}, None
        receiver_parameter = next(self.assets.base_model.parameters())
        for target, source, projector_index in routes:
            projector = self.assets.projector_list[projector_index]
            if not isinstance(projector, DirectPreRopeMLPProjector):
                raise TypeError("Direct MLP concat requires DirectPreRopeMLPProjector checkpoints.")
            key, value = projector.to(device=sharer_cache.key_cache[source].device, dtype=sharer_cache.key_cache[source].dtype).eval().project((sharer_cache.key_cache[source], sharer_cache.value_cache[source]))
            projected[target] = (key.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype), value.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype))
            prefix_length = int(key.shape[2])
        prefix = DynamicCache()
        for layer in range(int(self.assets.base_model.config.num_hidden_layers)):
            key, value = projected[layer]; prefix.key_cache.append(key.contiguous()); prefix.value_cache.append(value.contiguous())
        self.last_alignment_stats = {"alignment_type": "concat", "concat_projector_type": "direct_pre_rope_mlp", "routes": [list(route) for route in sorted(routes)]}
        return apply_receiver_compact_rope(self.assets.base_model, prefix)
