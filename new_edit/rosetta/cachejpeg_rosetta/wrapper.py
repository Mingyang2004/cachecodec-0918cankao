from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from rosetta.cachejpeg.wrapper import _ensure_homo_imports
from rosetta.cachejpeg.transport import build_transport
from rosetta.cachejpeg.transport import serialize_payload
from rosetta.model.latent_kv import (
    CacheAdapter,
    CacheJPEGLatentKVPayload,
    latent_payload_to_pseudo_kv_cache,
    pseudo_kv_cache_to_latent_payload,
)
from rosetta.model.projector import load_projector
from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    ProjectedKVAdaptiveQuantizer,
)
from rosetta.cachejpeg.packet import DCTInt16PacketCodec
from rosetta.cachejpeg.raw_dct_prune import RawDCTPrunePacketCodec
from rosetta.utils.evaluate import apply_generation_config, load_hf_model, set_default_chat_template
from rosetta.utils.timing import merge_present, new_timing_record, transport_fields

from .config import CacheJPEGRosettaEvalConfig, resolve_cachejpeg_rosetta_eval_config
from .cache_aligner import ConcatCacheAligner, RawConcatCacheAligner
from .direct_mlp_cache_aligner import DirectMLPConcatCacheAligner
from .projected_kv_cache_aligner import ProjectedKVConcatCacheAligner
from .fuser_bridge import LoadedRosettaAssets, RosettaFuserBridge
from .layer_streaming import LayerCompressionPipeline, LayerPrefillTimer, StreamingDynamicCache
from .concat_layer_streaming import ConcatLayerPipeline
from .adaptive_lcf_packet import AdaptiveLCFPacketCodec
from .pre_rope import (
    StreamingPreRopeDynamicCache,
    StreamingPreRopeKVPublisher,
    capture_pre_rope_keys,
    replace_cache_keys_with_pre_rope,
    stream_pre_rope_keys,
)


def _hf_local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


def _resolve_checkpoint_dir(checkpoints_dir: str, checkpoint_subfolder: Optional[str]) -> str:
    if not checkpoint_subfolder:
        return checkpoints_dir
    candidate = os.path.join(checkpoints_dir, checkpoint_subfolder)
    return candidate if os.path.isdir(candidate) else checkpoints_dir


def _load_projector_assets(checkpoint_dir: str) -> tuple[list[Any], dict[Any, Any]]:
    projector_list = []
    if os.path.isdir(checkpoint_dir):
        num_projectors = len(
            [f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)]
        )
        for proj_idx in range(num_projectors):
            json_cfg = os.path.join(checkpoint_dir, f"projector_{proj_idx}.json")
            pt_path = os.path.join(checkpoint_dir, f"projector_{proj_idx}.pt")
            proj = load_projector(json_cfg)
            state_dict = torch.load(pt_path, map_location="cpu")
            proj.load_state_dict(state_dict, strict=False)
            projector_list.append(proj)

        projector_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
        if os.path.isfile(projector_cfg_path):
            import json

            with open(projector_cfg_path, "r", encoding="utf-8") as f:
                projector_dict = json.load(f)
        else:
            projector_dict = {}
    else:
        projector_dict = {}
    return projector_list, projector_dict


def _load_cache_codec_assets(checkpoint_dir: str) -> list[Any]:
    codecs = []
    if not os.path.isdir(checkpoint_dir):
        return codecs
    indices = sorted(
        int(match.group(1))
        for name in os.listdir(checkpoint_dir)
        if (match := re.fullmatch(r"cache_codec_(\d+)\.pt", name))
    )
    for index in indices:
        config_path = os.path.join(checkpoint_dir, f"cache_codec_{index}.json")
        weight_path = os.path.join(checkpoint_dir, f"cache_codec_{index}.pt")
        codec = load_projector(config_path)
        codec.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
        codecs.append(codec)
    return codecs


def _load_rosetta_assets(
    model_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> LoadedRosettaAssets:
    rosetta_config = model_config.get("rosetta_config") or {}
    base_model_name = rosetta_config.get("base_model") or rosetta_config["receiver_model"]
    teacher_model_name = rosetta_config.get("teacher_model") or rosetta_config["sharer_model"]
    checkpoint_dir = _resolve_checkpoint_dir(
        rosetta_config["checkpoints_dir"],
        eval_config.get("rosetta_checkpoint_subfolder"),
    )

    base_model, base_tokenizer = load_hf_model(base_model_name, device=device, generation_config=generation_config)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=getattr(base_model, "dtype", None),
        local_files_only=_hf_local_files_only(),
    ).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        local_files_only=_hf_local_files_only(),
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    set_default_chat_template(teacher_tokenizer, str(teacher_model_name))
    apply_generation_config(teacher_model, generation_config)

    projector_list, projector_dict = _load_projector_assets(checkpoint_dir)
    cache_codec_list = _load_cache_codec_assets(checkpoint_dir)
    return LoadedRosettaAssets(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        projector_list=projector_list,
        projector_dict=projector_dict,
        checkpoint_dir=checkpoint_dir,
        cache_codec_list=cache_codec_list,
    )


class CacheJPEGRosettaEvalWrapper:
    """
    Skeleton for:
      teacher/sharer prefill -> cachejpeg encode/decode -> fuser -> base/receiver generate.
    """

    def __init__(
        self,
        assets: LoadedRosettaAssets,
        codec_config: dict[str, Any],
    ):
        self.assets = assets
        self.base_model = assets.base_model
        self.base_tokenizer = assets.base_tokenizer
        self.teacher_model = assets.teacher_model
        self.teacher_tokenizer = assets.teacher_tokenizer
        self.cache_codec_list = list(getattr(assets, "cache_codec_list", None) or [])
        teacher_parameter = next(self.teacher_model.parameters())
        for cache_codec in self.cache_codec_list:
            cache_codec.to(device=teacher_parameter.device, dtype=teacher_parameter.dtype).eval()
        self.eval_codec_config: CacheJPEGRosettaEvalConfig = resolve_cachejpeg_rosetta_eval_config(codec_config)
        self.fusion_type = self.eval_codec_config.fusion_type
        self.cache_alignment = self.eval_codec_config.cache_alignment
        concat_projector_config = dict(codec_config.get("concat_projector") or {})
        self.concat_projector_type = str(
            concat_projector_config.get("type", "lcf_first")
        ).lower()
        if self.concat_projector_type not in {
            "raw",
            "direct",
            "lcf_first",
            "lcf_projected_kv",
            "jcb",
            "split_channel_fusion",
            "direct_pre_rope_mlp",
        }:
            raise ValueError(
                "cachejpeg_rosetta.concat_projector.type must be 'raw', 'lcf_first', "
                "'lcf_projected_kv', 'jcb', or 'split_channel_fusion'."
            )
        if self.concat_projector_type in {"raw", "direct"} and self.eval_codec_config.adaptive_quant_table.enabled:
            raise ValueError("Raw concat does not support adaptive quantization.")
        self.split_latent_cachejpeg_enabled = (
            self.eval_codec_config.split_latent_cachejpeg.enabled
        )
        split_latent_cachejpeg_raw = dict(
            codec_config.get("split_latent_cachejpeg") or {}
        )
        self.split_latent_codec_config = dict(
            split_latent_cachejpeg_raw.get("codec") or {}
        )
        self.codec_config = {
            **codec_config.get("codec", {}),
            "homo_c2c_kv_src": self.eval_codec_config.homo_c2c_kv_src,
        }
        codec_method = str(self.codec_config.get("method", "")).lower()
        self.jcb_raw_transport = codec_method in {"jcb_raw", "raw_latent"}
        self.raw_cache_transport = codec_method in {"raw", "raw_kv"}
        self.raw_dct_prune_codec = RawDCTPrunePacketCodec.from_config(
            self.codec_config.get("raw_dct_prune")
        )
        self.direct_mlp_concat = (
            self.cache_alignment == "concat"
            and self.concat_projector_type == "direct_pre_rope_mlp"
        )
        fixed_packet_cfg = dict(self.codec_config.get("fixed_jcb_packet") or {})
        self.fixed_jcb_packet_codec = None
        if codec_method in {
            "jcb_dct_int16",
            "fixed_jcb_dct_int16",
        } or fixed_packet_cfg.get("enabled", False):
            self.fixed_jcb_packet_codec = DCTInt16PacketCodec(
                beta=float(fixed_packet_cfg.get("beta", self.codec_config.get("beta", 1.0))),
                low=float(fixed_packet_cfg.get("low", 1.0)),
                high=float(fixed_packet_cfg.get("high", 8.0)),
                scale_mode=str(
                    fixed_packet_cfg.get(
                        "scale_mode",
                        "rms" if bool(fixed_packet_cfg.get("scale", False)) else "none",
                    )
                ),
                entropy_backend="zlib6",
                timing_mode=str(fixed_packet_cfg.get("timing_mode", self.codec_config.get("timing_mode", "profile"))),
            )
        if self.concat_projector_type in {"raw", "direct"} and self.fixed_jcb_packet_codec is not None:
            raise ValueError("Raw concat must use full KV transport without a JCB packet.")
        if self.raw_dct_prune_codec is not None and not (
            self.raw_cache_transport
            or (self.cache_alignment == "concat" and self.concat_projector_type in {"raw", "direct", "direct_pre_rope_mlp"})
        ):
            raise ValueError(
                "raw_dct_prune is an ablation for raw KV transport only; use "
                "codec.method='raw' with raw concat or raw C2C fusion."
            )
        if (
            self.cache_alignment == "fuser"
            and (self.jcb_raw_transport or self.fixed_jcb_packet_codec is not None)
            and not self.cache_codec_list
        ):
            raise ValueError(
                "Fusion JCB transport requires cache_codec_N checkpoints in the run directory."
            )
        adaptive_concat_enabled = (
            self.cache_alignment == "concat"
            and self.eval_codec_config.adaptive_quant_table.enabled
        )
        if adaptive_concat_enabled:
            self.codec = None
            self._to_legacy_cache = CacheAdapter.to_legacy

            def to_dynamic_cache(cache):
                if isinstance(cache, DynamicCache):
                    return cache
                return DynamicCache.from_legacy_cache(CacheAdapter.to_legacy(cache))

            self._to_dynamic_cache = to_dynamic_cache
        elif self.raw_cache_transport or self.jcb_raw_transport or (
            self.cache_alignment == "concat"
            and (
                self.concat_projector_type in {"raw", "direct"}
                or self.concat_projector_type == "direct_pre_rope_mlp"
                or self.jcb_raw_transport
            )
        ) or (self.fixed_jcb_packet_codec is not None and self.cache_codec_list):
            self.codec = None
            self._to_legacy_cache = CacheAdapter.to_legacy

            def to_dynamic_cache(cache):
                if isinstance(cache, DynamicCache):
                    return cache
                return DynamicCache.from_legacy_cache(CacheAdapter.to_legacy(cache))

            self._to_dynamic_cache = to_dynamic_cache
        elif self.fusion_type == "latent_kv_split":
            # Split mode transmits LatentKVPayload directly and therefore has no
            # dependency on the CacheJPEG/HomoC2C codec implementation.
            self.codec = None
            if self.split_latent_cachejpeg_enabled:
                codec_cls, _, _, _ = _ensure_homo_imports(
                    self.eval_codec_config.homo_c2c_kv_src
                )
                if self.eval_codec_config.split_latent_cachejpeg.codec.compute.backend == "gpu":
                    from rosetta.cachejpeg.gpu_codec import GPUCacheJPEGCodec

                    self.codec = GPUCacheJPEGCodec(
                        device=next(self.teacher_model.parameters()).device
                    )
                else:
                    self.codec = codec_cls()
            self._to_legacy_cache = CacheAdapter.to_legacy

            def to_dynamic_cache(cache):
                if isinstance(cache, DynamicCache):
                    return cache
                return DynamicCache.from_legacy_cache(CacheAdapter.to_legacy(cache))

            self._to_dynamic_cache = to_dynamic_cache
        else:
            (
                codec_cls,
                _codec_config_resolver,
                self._to_dynamic_cache,
                self._to_legacy_cache,
            ) = _ensure_homo_imports(self.eval_codec_config.homo_c2c_kv_src)
            if self.eval_codec_config.codec.compute.backend == "gpu":
                from rosetta.cachejpeg.gpu_codec import GPUCacheJPEGCodec

                self.codec = GPUCacheJPEGCodec(
                    device=next(self.teacher_model.parameters()).device
                )
            else:
                self.codec = codec_cls()
        adaptive_quant_table = None
        self.adaptive_lcf_packet_codec = None
        if self.eval_codec_config.adaptive_quant_table.enabled:
            base_config = self.base_model.config
            if self.cache_alignment == "concat":
                adaptive_quant_table = ProjectedKVAdaptiveQuantizer(
                    num_layers=int(base_config.num_hidden_layers),
                    config=self.eval_codec_config.adaptive_quant_table,
                )
            else:
                adaptive_quant_table = AdaptiveCoefficientQuantizer(
                    num_layers=int(base_config.num_hidden_layers),
                    num_kv_heads=int(
                        getattr(
                            self.teacher_model.config,
                            "num_key_value_heads",
                            self.teacher_model.config.num_attention_heads,
                        )
                    ),
                    config=self.eval_codec_config.adaptive_quant_table,
                )
            state_path = Path(
                (codec_config.get("adaptive_quant_table") or {}).get(
                    "checkpoint_path",
                    Path(assets.checkpoint_dir or "") / "adaptive_quant_table.pt",
                )
            )
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"Adaptive quantization-table checkpoint not found: {state_path}"
                )
            adaptive_quant_table.load_state_dict(
                torch.load(state_path, map_location="cpu")
            )
            adaptive_quant_table.to(next(self.base_model.parameters()).device).eval()
            if self.cache_alignment == "concat":
                self.adaptive_lcf_packet_codec = AdaptiveLCFPacketCodec(
                    adaptive_quant_table
                )
        self.fuser_bridge = RosettaFuserBridge(
            assets,
            adaptive_quant_table=adaptive_quant_table,
            cache_codec_list=self.cache_codec_list,
            cache_codec_packet=self.fixed_jcb_packet_codec,
        )
        self.concat_cache_aligner = None
        if self.cache_alignment == "concat":
            if self.concat_projector_type == "direct_pre_rope_mlp":
                aligner_class = DirectMLPConcatCacheAligner
            elif self.concat_projector_type in {"raw", "direct"}:
                aligner_class = RawConcatCacheAligner
            elif self.concat_projector_type in {"lcf_projected_kv", "jcb", "split_channel_fusion"}:
                aligner_class = ProjectedKVConcatCacheAligner
            else:
                aligner_class = ConcatCacheAligner
            self.concat_cache_aligner = aligner_class(assets)
        if self.cache_alignment == "concat" and not assets.projector_list:
            raise ValueError(
                "cache_alignment='concat' requires a concat projector checkpoint."
            )
        if self.cache_alignment == "concat":
            expected_concat_projector = {
                "raw": {"DirectConcatProjector"},
                "direct": {"DirectConcatProjector"},
                "lcf_first": {"LCFFirstProjector"},
                "lcf_projected_kv": {"LCFProjectedKVProjector", "JCBProjector"},
                "jcb": {"JCBProjector"},
                "split_channel_fusion": {"SplitChannelFusionProjector"},
                "direct_pre_rope_mlp": {"DirectPreRopeMLPProjector"},
            }[self.concat_projector_type]
            unexpected = sorted(
                {
                    projector.__class__.__name__
                    for projector in assets.projector_list
                    if projector.__class__.__name__ not in expected_concat_projector
                }
            )
            if unexpected:
                raise ValueError(
                    f"concat_projector.type={self.concat_projector_type!r} requires "
                    f"{sorted(expected_concat_projector)}, "
                    f"but loaded {unexpected}."
                )
        transport_config = codec_config.get("transport") or (codec_config.get("codec") or {}).get("transport")
        self.transport = build_transport(dict(transport_config or {}))
        self.last_transport_stats = None
        self.last_codec_stats: dict[str, Any] | None = None
        # Kept separate from legacy ``last_codec_stats``.  Evaluator code must
        # use this record for timing summaries; codec stats remain for size and
        # compatibility reporting.
        self.last_timing_stats: dict[str, Any] | None = None
        self._packet_timing_totals: dict[str, float] = {}
        self.last_fusion_stats: dict[str, Any] | None = None
        self.ablation_config = dict(codec_config.get("ablation") or {})
        if self.fusion_type in {"latent_kv_joint", "latent_kv_split"}:
            if not assets.projector_list:
                raise ValueError(
                    f"fusion_type={self.fusion_type!r} requires a compatible "
                    "latent KV checkpoint."
                )
            expected_class = (
                "LatentKVCompressor"
                if self.fusion_type == "latent_kv_joint"
                else "SplitLatentKVProjector"
            )
            unexpected = [
                projector.__class__.__name__
                for projector in assets.projector_list
                if projector.__class__.__name__ != expected_class
            ]
            if unexpected:
                raise ValueError(
                    f"fusion_type={self.fusion_type!r} requires {expected_class}, "
                    f"but loaded {sorted(set(unexpected))}."
                )
        if self.eval_codec_config.layer_streaming.enabled and not hasattr(self.codec, "encode_layer"):
            raise ValueError(
                "cachejpeg_rosetta.layer_streaming currently requires compute.backend=gpu."
            )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @staticmethod
    def _split_dual_inputs(input_ids, attention_mask=None):
        """Split independently tokenized receiver and sharer inputs."""
        if isinstance(input_ids, (list, tuple)):
            if len(input_ids) != 2:
                raise ValueError(
                    "CacheJPEG-Rosetta inputs must contain receiver and sharer tensors"
                )
            base_input_ids, teacher_input_ids = input_ids
            if isinstance(attention_mask, (list, tuple)):
                if len(attention_mask) != 2:
                    raise ValueError(
                        "CacheJPEG-Rosetta attention_mask must contain receiver and sharer tensors"
                    )
                base_attention_mask, teacher_attention_mask = attention_mask
            else:
                base_attention_mask = teacher_attention_mask = attention_mask
            return base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask
        return input_ids, input_ids, attention_mask, attention_mask

    def _configured_frequency_prune_stats(self) -> dict[str, Any]:
        eval_cfg = getattr(self, "eval_codec_config", None)
        codec_eval_cfg = getattr(eval_cfg, "codec", None)
        prune_eval_cfg = getattr(codec_eval_cfg, "frequency_prune", None)
        if prune_eval_cfg is not None:
            return {
                "enabled": bool(prune_eval_cfg.enabled),
                "prune_from": str(prune_eval_cfg.prune_from),
            }
        raw_prune = (getattr(self, "codec_config", {}) or {}).get(
            "frequency_prune", {}
        )
        return {
            "enabled": bool(raw_prune.get("enabled", False)),
            "prune_from": str(
                raw_prune.get("prune_from", raw_prune.get("prune_from_band", "B4"))
            ).upper(),
        }

    def _decode_from_receiver_cache(
        self,
        last_token: torch.Tensor,
        past_key_values,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> torch.Tensor:
        generated = []
        current_input = last_token
        current_past = self._to_dynamic_cache(past_key_values)
        model_device = next(self.base_model.parameters()).device
        if model_device.type == "cuda":
            torch.cuda.synchronize(model_device)
        decode_started = time.perf_counter()
        first_token_ms = None
        for _ in range(max(1, int(max_new_tokens))):
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=current_input,
                    past_key_values=current_past,
                    use_cache=True,
                )
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            logits = outputs.logits[:, -1, :]
            if do_sample:
                scaled = logits if temperature <= 0 else logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - decode_started) * 1000.0
            generated.append(next_token)
            current_input = next_token
            current_past = self._to_dynamic_cache(outputs.past_key_values)
            if self.base_tokenizer.eos_token_id is not None and next_token.item() == self.base_tokenizer.eos_token_id:
                break
        self._last_receiver_decode_timing = {
            "receiver_first_token_ms": float(first_token_ms or 0.0),
            "receiver_decode_ms": float((time.perf_counter() - decode_started) * 1000.0),
        }
        if not generated:
            return last_token
        return torch.cat(generated, dim=1)

    def prefill_on_sharer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        *,
        pre_rope: bool = False,
    ):
        model_kwargs = {}
        if past_key_values is not None:
            model_kwargs["past_key_values"] = past_key_values
        if pre_rope:
            if past_key_values is not None:
                raise ValueError("pre-RoPE sharer capture does not support an existing cache.")
            with capture_pre_rope_keys(self.teacher_model) as captured:
                with torch.no_grad():
                    outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
            outputs.past_key_values = replace_cache_keys_with_pre_rope(
                outputs.past_key_values, captured
            )
            return outputs
        with torch.no_grad():
            return self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **model_kwargs,
            )

    def prefill_on_receiver(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

    def encode_cache(self, past_key_values):
        legacy_cache = self._to_legacy_cache(past_key_values)
        if not getattr(self.codec, "uses_gpu_transform", False):
            legacy_cache = self._cache_to_codec_dtype(legacy_cache)
        return self.codec.encode(legacy_cache, self.codec_config)

    def decode_cache(self, payload):
        return self.codec.decode(payload, self.codec_config)

    def _encode_fixed_jcb_packet(self, latent_cache):
        if self.fixed_jcb_packet_codec is None:
            raise RuntimeError("Fixed JCB packet codec is not configured.")
        self._packet_timing_totals = {}
        layers = []
        for key, value in latent_cache:
            encoded_key = self.fixed_jcb_packet_codec.encode(key)
            self._accumulate_packet_timing("encode")
            encoded_value = self.fixed_jcb_packet_codec.encode(value)
            self._accumulate_packet_timing("encode")
            layers.append({"key": encoded_key, "value": encoded_value})
        return {"method": "jcb_dct_int16", "version": 1, "layers": layers}

    def _decode_fixed_jcb_packet(self, packet, device):
        if self.fixed_jcb_packet_codec is None:
            raise RuntimeError("Fixed JCB packet codec is not configured.")
        if packet.get("method") != "jcb_dct_int16" or packet.get("version") != 1:
            raise ValueError("Unsupported fixed JCB packet.")
        decoded = []
        for layer in packet["layers"]:
            key = self.fixed_jcb_packet_codec.decode(layer["key"], device=device)
            self._accumulate_packet_timing("decode")
            value = self.fixed_jcb_packet_codec.decode(layer["value"], device=device)
            self._accumulate_packet_timing("decode")
            decoded.append((key, value))
        return tuple(decoded)

    def _accumulate_packet_timing(self, direction: str) -> None:
        """Accumulate per-tensor DCT packet stages for the current request."""
        stats = getattr(self.fixed_jcb_packet_codec, f"last_{direction}_stats", None) or {}
        for name, value in stats.items():
            if isinstance(value, (int, float)):
                self._packet_timing_totals[name] = self._packet_timing_totals.get(name, 0.0) + float(value)

    def _finalize_timing(self, output: torch.Tensor, *, route: str, execution_mode: str = "serial") -> torch.Tensor:
        """Publish a stable timing record after every CacheJPEG generation path.

        ``model_e2e_ms`` in the evaluator remains authoritative for comparisons
        because it surrounds this whole call.  This internal value is retained
        only for auditing component coverage.
        """
        legacy = self.last_codec_stats or {}
        quantized = getattr(self, "fixed_jcb_packet_codec", None) is not None
        bridge = "jcb" if (quantized or getattr(self, "cache_codec_list", None) or getattr(self, "concat_projector_type", "") == "jcb") else "none"
        record = new_timing_record(
            route=route,
            bridge=bridge,
            codec="dct_int16_zlib6" if quantized else "raw",
            execution_mode=execution_mode,
        )
        merge_present(record, transport_fields(self.last_transport_stats))
        packet = self._packet_timing_totals
        merge_present(record, {
            "sharer_prefill_ms": legacy.get("sharer_prefill_ms"),
            "sender_projection_ms": legacy.get("sender_encode_ms", legacy.get("lcf_encode_seconds", 0.0) * 1000.0 if legacy.get("lcf_encode_seconds") is not None else None),
            "gpu_dct_ms": packet.get("gpu_dct_ms"),
            "gpu_quant_ms": packet.get("gpu_quant_ms"),
            "d2h_ms": packet.get("d2h_ms"),
            "entropy_encode_ms": packet.get("entropy_encode_ms"),
            "entropy_decode_ms": packet.get("entropy_decode_ms"),
            "h2d_ms": packet.get("h2d_ms"),
            "gpu_idct_ms": packet.get("gpu_idct_ms"),
            "receiver_projection_or_fusion_ms": legacy.get("fusion_ms", legacy.get("receiver_decode_ms")),
            "receiver_prefill_ms": legacy.get("receiver_prefill_ms"),
            "receiver_first_token_ms": legacy.get("receiver_first_token_ms"),
            "receiver_decode_ms": legacy.get("receiver_decode_ms"),
            "ttft_ms": legacy.get("ttft_ms"),
        })
        # For serial routes this is deliberately a component audit, not an
        # alternative wall-clock e2e measurement.  Streaming paths replace it
        # with their pipeline critical-path value below when available.
        record["pipeline_critical_path_ms"] = (
            legacy.get("pipeline_critical_path_ms")
            if legacy.get("pipeline_critical_path_ms") is not None
            else (
                float(legacy["pipeline_seconds"]) * 1000.0
                if legacy.get("pipeline_seconds") is not None
                else None
            )
        )
        record["model_e2e_internal_ms"] = legacy.get("internal_end_to_end_latency_ms")
        self.last_timing_stats = record
        return output

    def _encode_c2c_source_packet(self, source_cache):
        if not self.cache_codec_list:
            return source_cache
        self._packet_timing_totals = {}
        layers = []
        projector_dict = getattr(self.fuser_bridge, "projector_dict", {})
        layer_map = projector_dict.get(
            int(self.assets.base_model_idx), {}
        ).get(int(self.assets.teacher_model_idx), {})
        for target_layer, entry in layer_map.items():
            pairs = entry if isinstance(entry, list) else [entry]
            if not pairs:
                continue
            source_layer, _ = self.fuser_bridge._normalize_pair(pairs[0])
            codec = self.cache_codec_list[min(int(target_layer), len(self.cache_codec_list) - 1)]
            key_latent, value_latent = codec.encode((source_cache[source_layer][0], source_cache[source_layer][1]))
            fixed_packet_codec = getattr(self, "fixed_jcb_packet_codec", None)
            if fixed_packet_codec is not None:
                key_latent = fixed_packet_codec.encode(key_latent.unsqueeze(1))
                self._accumulate_packet_timing("encode")
                value_latent = fixed_packet_codec.encode(value_latent.unsqueeze(1))
                self._accumulate_packet_timing("encode")
            else:
                key_latent = key_latent.detach().cpu().contiguous()
                value_latent = value_latent.detach().cpu().contiguous()
            layers.append((int(source_layer), int(target_layer), key_latent, value_latent))
        return {"method": "c2c_jcb_source", "version": 1, "layers": layers}

    def _decode_c2c_source_packet(self, packet, source_cache):
        if not self.cache_codec_list:
            return source_cache
        if packet.get("method") != "c2c_jcb_source":
            raise ValueError("Unsupported C2C JCB source packet.")
        restored = list(source_cache)
        for source_layer, target_layer, key_data, value_data in packet["layers"]:
            codec = self.cache_codec_list[min(int(target_layer), len(self.cache_codec_list) - 1)]
            fixed_packet_codec = getattr(self, "fixed_jcb_packet_codec", None)
            if fixed_packet_codec is not None:
                key_latent = fixed_packet_codec.decode(key_data, device=source_cache[source_layer][0].device).squeeze(1)
                self._accumulate_packet_timing("decode")
                value_latent = fixed_packet_codec.decode(value_data, device=source_cache[source_layer][1].device).squeeze(1)
                self._accumulate_packet_timing("decode")
            else:
                key_latent = key_data.to(
                    device=source_cache[source_layer][0].device,
                    dtype=source_cache[source_layer][0].dtype,
                )
                value_latent = value_data.to(
                    device=source_cache[source_layer][1].device,
                    dtype=source_cache[source_layer][1].dtype,
                )
            restored[int(source_layer)] = codec.decode_source(key_latent, value_latent)
        return tuple(restored)

    @staticmethod
    def _cache_to_codec_dtype(past_key_values):
        return tuple(
            (
                key.detach().to(dtype=torch.float32),
                value.detach().to(dtype=torch.float32),
            )
            for key, value in past_key_values
        )

    def fuse_to_receiver_cache(self, decoded_teacher_cache, base_seed_cache=None, *, apply_cache_codec=True):
        try:
            fused = self.fuser_bridge.fuse_teacher_cache_to_base(
                decoded_teacher_cache,
                base_seed_cache=base_seed_cache,
                apply_cache_codec=apply_cache_codec,
            )
        except TypeError as error:
            if "apply_cache_codec" not in str(error):
                raise
            fused = self.fuser_bridge.fuse_teacher_cache_to_base(
                decoded_teacher_cache,
                base_seed_cache=base_seed_cache,
            )
        self.last_fusion_stats = getattr(
            self.fuser_bridge, "last_fusion_stats", None
        )
        return fused

    def generate_on_receiver(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

    def _generate_with_layer_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_dual_inputs(input_ids, attention_mask)
        )
        num_layers = int(self.teacher_model.config.num_hidden_layers)
        pipeline = LayerCompressionPipeline(
            codec=self.codec,
            codec_config=self.codec_config,
            transport=self.transport,
            num_layers=num_layers,
            queue_size=self.eval_codec_config.layer_streaming.queue_size,
        )
        streaming_cache = StreamingDynamicCache(pipeline.submit)
        prefill_timer = LayerPrefillTimer(self.teacher_model, num_layers)
        prefill_timer.start()
        pipeline_started = time.perf_counter()
        try:
            self.prefill_on_sharer(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                past_key_values=streaming_cache,
            )
            # Receiver prefill overlaps with any compression work still queued.
            receiver_seed_outputs = self.prefill_on_receiver(
                input_ids=base_input_ids, attention_mask=base_attention_mask
            )
            decoded_teacher_cache = pipeline.finish()
            teacher_device = next(self.teacher_model.parameters()).device
            if teacher_device.type == "cuda":
                torch.cuda.synchronize(teacher_device)
            layer_prefill_seconds = prefill_timer.finish()
        except BaseException:
            for handle in prefill_timer.handles:
                handle.remove()
            prefill_timer.handles.clear()
            pipeline.abort()
            raise
        pipeline_seconds = time.perf_counter() - pipeline_started
        self.last_transport_stats = pipeline.aggregate_transport_stats()

        zero_sharer_cache = bool(self.ablation_config.get("zero_sharer_cache_at_receiver", False))
        if zero_sharer_cache:
            decoded_teacher_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in decoded_teacher_cache
            )
        self.last_codec_stats = {
            "mode": "layer_streaming",
            "num_layers": num_layers,
            "queue_size": self.eval_codec_config.layer_streaming.queue_size,
            "frequency_prune": self._configured_frequency_prune_stats(),
            "original_kv_bytes": pipeline.original_kv_bytes,
            "payload_bytes": pipeline.payload_bytes,
            "compression_factor": (
                float(pipeline.original_kv_bytes / pipeline.payload_bytes)
                if pipeline.payload_bytes
                else 0.0
            ),
            "space_saving_ratio": (
                float(1.0 - pipeline.payload_bytes / pipeline.original_kv_bytes)
                if pipeline.original_kv_bytes
                else 0.0
            ),
            "encode_seconds": float(pipeline.encode_seconds),
            "avg_layer_encode_seconds": float(
                sum(value for value in pipeline.layer_encode_seconds if value is not None)
                / max(1, sum(value is not None for value in pipeline.layer_encode_seconds))
            ),
            "layer_encode_seconds": [
                float(value) if value is not None else None
                for value in pipeline.layer_encode_seconds
            ],
            "avg_layer_prefill_seconds": float(sum(layer_prefill_seconds) / len(layer_prefill_seconds)),
            "layer_prefill_seconds": [float(value) for value in layer_prefill_seconds],
            "decode_seconds": float(pipeline.decode_seconds),
            "pipeline_seconds": float(pipeline_seconds),
            "transport_bandwidth_bytes_per_sec": getattr(
                self.transport, "bandwidth_bytes_per_sec", None
            ),
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
        }
        fused_receiver_cache = self.fuse_to_receiver_cache(
            decoded_teacher_cache,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([base_input_ids, generated], dim=1)

    def _generate_with_split_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        """Transmit only sharer-produced latent tensors, then fuse at receiver."""

        # Keep compatibility with lightweight wrappers created via ``__new__``
        # in callers that predate the optional CacheJPEG path.
        split_latent_cachejpeg_enabled = bool(
            getattr(self, "split_latent_cachejpeg_enabled", False)
        )
        fixed_packet_enabled = getattr(self, "fixed_jcb_packet_codec", None) is not None

        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_dual_inputs(input_ids, attention_mask)
        )
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        sharer_prefill_started = time.perf_counter()
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        sharer_prefill_ms = (time.perf_counter() - sharer_prefill_started) * 1000.0
        receiver_device = next(self.base_model.parameters()).device
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        receiver_prefill_started = time.perf_counter()
        receiver_seed_outputs = self.prefill_on_receiver(
            input_ids=base_input_ids, attention_mask=base_attention_mask
        )
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        receiver_prefill_ms = (time.perf_counter() - receiver_prefill_started) * 1000.0
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        original_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in legacy_sharer_cache
        )

        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_started = time.perf_counter()
        latent_payload = self.fuser_bridge.encode_teacher_cache_to_latents(
            sharer_outputs.past_key_values,
            move_to_cpu=not (split_latent_cachejpeg_enabled or fixed_packet_enabled),
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_seconds = time.perf_counter() - encode_started

        latent_bytes = sum(
            int(layer.latent.numel() * layer.latent.element_size())
            for layer in latent_payload.layers
        )
        cachejpeg_encode_seconds = 0.0
        wire_payload = latent_payload
        cachejpeg_summary = None
        if split_latent_cachejpeg_enabled or fixed_packet_enabled:
            pseudo_cache = latent_payload_to_pseudo_kv_cache(latent_payload)
            if not fixed_packet_enabled and not getattr(self.codec, "uses_gpu_transform", False):
                pseudo_cache = self._cache_to_codec_dtype(pseudo_cache)
            if teacher_device.type == "cuda":
                torch.cuda.synchronize(teacher_device)
            cachejpeg_encode_started = time.perf_counter()
            encoded_payload = (
                self._encode_fixed_jcb_packet(pseudo_cache)
                if fixed_packet_enabled
                else self.codec.encode(pseudo_cache, self.split_latent_codec_config)
            )
            if teacher_device.type == "cuda":
                torch.cuda.synchronize(teacher_device)
            cachejpeg_encode_seconds = (
                time.perf_counter() - cachejpeg_encode_started
            )
            cachejpeg_summary = dict(getattr(encoded_payload, "local_summary", {}) or {})
            wire_payload = CacheJPEGLatentKVPayload(
                encoded_payload=encoded_payload,
                layers=[
                    (
                        int(layer.receiver_layer),
                        int(layer.sharer_layer),
                        int(layer.projector_idx),
                    )
                    for layer in latent_payload.layers
                ],
                latent_dim=latent_payload.latent_dim,
                sequence_length=latent_payload.sequence_length,
                source_dtype=latent_payload.source_dtype,
                entropy_backend=(
                    "zlib6" if fixed_packet_enabled else self.eval_codec_config.split_latent_cachejpeg.codec.entropy.backend
                ),
            )

        transport = getattr(self, "transport", None)
        received_payload = (
            transport.roundtrip(wire_payload)
            if transport is not None
            else wire_payload
        )
        self.last_transport_stats = (
            transport.last_stats if transport is not None else None
        )
        payload_bytes = int(getattr(self.last_transport_stats, "frame_bytes", 0)) or len(serialize_payload(wire_payload))

        cachejpeg_decode_seconds = 0.0
        decoder_payload = received_payload
        receiver_device = next(self.base_model.parameters()).device
        if split_latent_cachejpeg_enabled or fixed_packet_enabled:
            if not isinstance(received_payload, CacheJPEGLatentKVPayload):
                raise TypeError(
                    "Expected CacheJPEGLatentKVPayload after split transport, got "
                    f"{type(received_payload)!r}."
                )
            if receiver_device.type == "cuda":
                torch.cuda.synchronize(receiver_device)
            cachejpeg_decode_started = time.perf_counter()
            reconstructed_pseudo_cache = (
                self._decode_fixed_jcb_packet(
                    received_payload.encoded_payload,
                    device=receiver_device,
                )
                if fixed_packet_enabled
                else self.codec.decode(
                    received_payload.encoded_payload,
                    self.split_latent_codec_config,
                )
            )
            if receiver_device.type == "cuda":
                torch.cuda.synchronize(receiver_device)
            cachejpeg_decode_seconds = (
                time.perf_counter() - cachejpeg_decode_started
            )
            decoder_payload = pseudo_kv_cache_to_latent_payload(
                reconstructed_pseudo_cache, received_payload
            )

        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        decode_started = time.perf_counter()
        fused_receiver_cache = self.fuser_bridge.fuse_latents_to_base(
            decoder_payload,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        decode_seconds = time.perf_counter() - decode_started
        self.last_fusion_stats = self.fuser_bridge.last_fusion_stats
        self.last_codec_stats = {
            "mode": (
                "latent_kv_split_cachejpeg"
                if split_latent_cachejpeg_enabled or fixed_packet_enabled
                else "latent_kv_split"
            ),
            "quantized": split_latent_cachejpeg_enabled or fixed_packet_enabled,
            "entropy_backend": (
                self.eval_codec_config.split_latent_cachejpeg.codec.entropy.backend
                if split_latent_cachejpeg_enabled or fixed_packet_enabled
                else None
            ),
            "original_kv_bytes": original_kv_bytes,
            "latent_bytes": latent_bytes,
            "metadata_bytes": (
                None
                if split_latent_cachejpeg_enabled or fixed_packet_enabled
                else max(0, payload_bytes - latent_bytes)
            ),
            "payload_bytes": payload_bytes,
            "compression_factor": (
                float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0
            ),
            "latent_compression_factor": (
                float(latent_bytes / payload_bytes) if payload_bytes else 0.0
            ),
            "latent_element_compression_factor": (
                float(
                    sum(key.numel() + value.numel() for key, value in legacy_sharer_cache)
                    / sum(layer.latent.numel() for layer in latent_payload.layers)
                )
                if latent_payload.layers
                else 0.0
            ),
            "encode_seconds": float(
                encode_seconds + cachejpeg_encode_seconds
            ),
            "decode_seconds": float(
                cachejpeg_decode_seconds + decode_seconds
            ),
            "latent_encode_seconds": float(encode_seconds),
            "cachejpeg_encode_seconds": float(cachejpeg_encode_seconds),
            "cachejpeg_decode_seconds": float(cachejpeg_decode_seconds),
            "receiver_decode_seconds": float(decode_seconds),
            "cachejpeg_summary": cachejpeg_summary,
        }
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([base_input_ids, generated], dim=1)

    def _decode_from_prefill_outputs(
        self,
        prefill_outputs,
        *,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> torch.Tensor:
        """Decode without feeding the receiver prompt's final token a second time."""

        generated = []
        current_past = self._to_dynamic_cache(prefill_outputs.past_key_values)
        current_logits = prefill_outputs.logits[:, -1, :]
        current_attention_mask = attention_mask
        eos_token_id = self.base_tokenizer.eos_token_id
        model_device = next(self.base_model.parameters()).device
        if model_device.type == "cuda":
            torch.cuda.synchronize(model_device)
        decode_started = time.perf_counter()
        first_token_ms = None
        for generated_index in range(max(1, int(max_new_tokens))):
            if do_sample:
                scaled = current_logits if temperature <= 0 else current_logits / temperature
                next_token = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(current_logits, dim=-1, keepdim=True)
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - decode_started) * 1000.0
            generated.append(next_token)
            if eos_token_id is not None and bool(torch.all(next_token == int(eos_token_id))):
                break
            if generated_index + 1 == max(1, int(max_new_tokens)):
                break
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=next_token,
                    attention_mask=current_attention_mask,
                    past_key_values=current_past,
                    use_cache=True,
                )
            current_past = self._to_dynamic_cache(outputs.past_key_values)
            current_logits = outputs.logits[:, -1, :]
        self._last_receiver_decode_timing = {
            "receiver_first_token_ms": float(first_token_ms or 0.0),
            "receiver_decode_ms": float((time.perf_counter() - decode_started) * 1000.0),
        }
        return torch.cat(generated, dim=1)

    def _generate_with_direct_mlp_concat(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], generation_config: dict[str, Any]
    ) -> torch.Tensor:
        """Raw pre-RoPE KV transport -> optional DCT prune -> historical direct MLP."""
        base_ids, teacher_ids, base_mask, teacher_mask = self._split_dual_inputs(input_ids, attention_mask)
        sharer_outputs = self.prefill_on_sharer(input_ids=teacher_ids, attention_mask=teacher_mask, pre_rope=True)
        source_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        original_kv_bytes = sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in source_cache)
        if not isinstance(self.concat_cache_aligner, DirectMLPConcatCacheAligner):
            raise RuntimeError("Direct MLP concat aligner was not initialized.")
        if self.raw_dct_prune_codec is None:
            raise RuntimeError("Direct MLP DCT ablation requires raw_dct_prune configuration.")
        encode_started = time.perf_counter()
        payload = self.raw_dct_prune_codec.encode(source_cache)
        encode_seconds = time.perf_counter() - encode_started
        received = self.transport.roundtrip(payload) if self.transport is not None else payload
        self.last_transport_stats = self.transport.last_stats if self.transport is not None else None
        payload_bytes = int(getattr(self.last_transport_stats, "frame_bytes", 0)) or len(serialize_payload(payload))
        decode_started = time.perf_counter()
        restored_cache = self.raw_dct_prune_codec.decode(received, source_cache)
        decode_seconds = time.perf_counter() - decode_started
        prefix = self.concat_cache_aligner.align(restored_cache)
        self.last_fusion_stats = self.concat_cache_aligner.last_alignment_stats
        prefix_length = int(prefix.key_cache[0].shape[2])
        if base_mask is None: base_mask = torch.ones_like(base_ids, dtype=torch.long)
        prefix_mask = torch.ones((base_ids.shape[0], prefix_length), dtype=base_mask.dtype, device=base_mask.device)
        combined_mask = torch.cat((prefix_mask, base_mask), dim=1)
        position_ids = (base_mask.long().cumsum(-1) - 1 + prefix_length).masked_fill(base_mask == 0, 0)
        cache_position = torch.arange(prefix_length, prefix_length + base_ids.shape[1], device=base_ids.device)
        with torch.no_grad():
            outputs = self.base_model(input_ids=base_ids, attention_mask=combined_mask, position_ids=position_ids,
                                      cache_position=cache_position, past_key_values=prefix, use_cache=True)
        generated = self._decode_from_prefill_outputs(outputs, attention_mask=combined_mask,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)), temperature=float(generation_config.get("temperature", 0.0)))
        self.last_codec_stats = {
            "original_kv_bytes": int(original_kv_bytes), "payload_bytes": int(payload_bytes),
            "compression_factor": float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0,
            "encode_seconds": float(encode_seconds), "decode_seconds": float(decode_seconds),
            "cache_alignment": "concat", "concat_projector_type": "direct_pre_rope_mlp",
            "codec_order": "raw_kv_dct_prune_idct_direct_mlp", "quantization_mode": "none",
            "raw_dct_prune": dict(self.raw_dct_prune_codec.last_stats or {}), "prefix_tokens": prefix_length,
        }
        return torch.cat([base_ids, generated], dim=1)

    def _generate_with_concat_alignment(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        """Use mapped Sharer pre-RoPE KV as a prefix to Receiver prefill and decode."""

        base_ids, teacher_ids, base_mask, teacher_mask = self._split_dual_inputs(
            input_ids, attention_mask
        )
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        sharer_prefill_started = time.perf_counter()
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            pre_rope=True,
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        sharer_prefill_ms = (time.perf_counter() - sharer_prefill_started) * 1000.0
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        if self.concat_cache_aligner is None:
            raise RuntimeError("Concat cache aligner was not initialized.")
        lcf_encode_started = time.perf_counter()
        lcf_latent_cache, lcf_routing = self.concat_cache_aligner.encode(
            legacy_sharer_cache
        )
        original_kv_bytes = sum(
            int(
                legacy_sharer_cache[source_layer][0].numel()
                * legacy_sharer_cache[source_layer][0].element_size()
                + legacy_sharer_cache[source_layer][1].numel()
                * legacy_sharer_cache[source_layer][1].element_size()
            )
            for _target_layer, source_layer, _projector_index in lcf_routing.routes
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        lcf_encode_seconds = time.perf_counter() - lcf_encode_started
        latent_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in lcf_latent_cache
        )
        raw_concat = self.concat_projector_type in {"raw", "direct"}
        raw_latent = bool(self.jcb_raw_transport and not raw_concat)
        encode_started = time.perf_counter()
        if raw_concat or raw_latent:
            if self.raw_dct_prune_codec is not None:
                payload = self.raw_dct_prune_codec.encode(lcf_latent_cache)
            else:
                payload = tuple(
                    (
                        key.detach().cpu().contiguous(),
                        value.detach().cpu().contiguous(),
                    )
                    for key, value in lcf_latent_cache
                )
        elif self.fixed_jcb_packet_codec is not None:
            payload = self._encode_fixed_jcb_packet(lcf_latent_cache)
        elif self.adaptive_lcf_packet_codec is not None:
            payload = self.adaptive_lcf_packet_codec.encode(lcf_latent_cache)
        else:
            payload = self.encode_cache(lcf_latent_cache)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_seconds = time.perf_counter() - encode_started
        received_payload = (
            self.transport.roundtrip(payload) if self.transport is not None else payload
        )
        self.last_transport_stats = (
            self.transport.last_stats if self.transport is not None else None
        )
        payload_bytes = int(getattr(self.last_transport_stats, "frame_bytes", 0)) or len(serialize_payload(payload))
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_started = time.perf_counter()
        if raw_concat or raw_latent:
            decoded_latent_cache = (
                self.raw_dct_prune_codec.decode(received_payload, lcf_latent_cache)
                if self.raw_dct_prune_codec is not None
                else received_payload
            )
        elif self.fixed_jcb_packet_codec is not None:
            decoded_latent_cache = self._decode_fixed_jcb_packet(
                received_payload,
                device=next(self.base_model.parameters()).device,
            )
        elif self.adaptive_lcf_packet_codec is not None:
            decoded_latent_cache = self.adaptive_lcf_packet_codec.decode(
                received_payload,
                device=next(self.base_model.parameters()).device,
            ).past_key_values
        else:
            decoded_latent_cache = self.decode_cache(received_payload)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_seconds = time.perf_counter() - decode_started

        if bool(self.ablation_config.get("zero_sharer_cache_at_receiver", False)):
            decoded_latent_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in self._to_legacy_cache(decoded_latent_cache)
            )
        lcf_decode_started = time.perf_counter()
        receiver_prefix = self.concat_cache_aligner.decode(
            decoded_latent_cache, lcf_routing
        )
        receiver_device = next(self.base_model.parameters()).device
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        lcf_decode_seconds = time.perf_counter() - lcf_decode_started
        self.last_fusion_stats = self.concat_cache_aligner.last_alignment_stats
        prefix_length = int(receiver_prefix.key_cache[0].shape[2])

        if base_mask is None:
            base_mask = torch.ones_like(base_ids, dtype=torch.long)
        prefix_mask = torch.ones(
            (base_ids.shape[0], prefix_length),
            dtype=base_mask.dtype,
            device=base_mask.device,
        )
        combined_mask = torch.cat([prefix_mask, base_mask], dim=1)
        receiver_position_ids = base_mask.long().cumsum(-1) - 1 + prefix_length
        receiver_position_ids = receiver_position_ids.masked_fill(base_mask == 0, 0)
        receiver_device = next(self.base_model.parameters()).device
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        receiver_prefill_started = time.perf_counter()
        with torch.no_grad():
            receiver_outputs = self.base_model(
                input_ids=base_ids,
                attention_mask=combined_mask,
                position_ids=receiver_position_ids,
                past_key_values=receiver_prefix,
                use_cache=True,
            )
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        receiver_prefill_ms = (time.perf_counter() - receiver_prefill_started) * 1000.0
        generated = self._decode_from_prefill_outputs(
            receiver_outputs,
            attention_mask=combined_mask,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        self.last_codec_stats = {
            "original_kv_bytes": original_kv_bytes,
            "lcf_latent_kv_bytes": latent_kv_bytes,
            "payload_bytes": payload_bytes,
            "compression_factor": float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0,
            "space_saving_ratio": float(1.0 - payload_bytes / original_kv_bytes) if original_kv_bytes else 0.0,
            "encode_seconds": float(encode_seconds),
            "decode_seconds": float(decode_seconds),
            "lcf_encode_seconds": float(lcf_encode_seconds),
            "lcf_decode_seconds": float(lcf_decode_seconds),
            "sender_encode_seconds": float(lcf_encode_seconds + encode_seconds),
            "receiver_decode_seconds": float(decode_seconds + lcf_decode_seconds),
            "compute_backend": self.eval_codec_config.codec.compute.backend,
            "transform_dtype": self.eval_codec_config.codec.compute.transform_dtype,
            "entropy_backend": self.eval_codec_config.codec.entropy.backend,
            "layer_streaming": False,
            "layer_execution": "whole_cache_sequential",
            "transport_mode": self.eval_codec_config.codec.transport.mode,
            "transport_bandwidth_bytes_per_sec": (
                self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
            ),
            "bandwidth_only_transmit_seconds": (
                float(
                    payload_bytes
                    / self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
                )
                if self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
                else None
            ),
            "cache_alignment": "concat",
            "concat_projector_type": self.concat_projector_type,
            "raw_dct_prune": (
                dict(self.raw_dct_prune_codec.last_stats or {})
                if self.raw_dct_prune_codec is not None
                else {"enabled": False}
            ),
            "codec_order": (
                "raw_kv_dct_prune_idct_direct_concat"
                if raw_concat and self.raw_dct_prune_codec is not None
                else ("raw_transport_direct_concat"
                if raw_concat
                else (
                    "jcb_raw_latent_transport"
                    if raw_latent
                    else (
                        "lcf_project_kv_cachejpeg_lcf_up"
                        if self.concat_projector_type == "lcf_projected_kv"
                        else "lcf_down_cachejpeg_lcf_up"
                    )
                ))
            ),
            "rope_mode": "pre_rope",
            "prefix_tokens": prefix_length,
            "latent_dim": lcf_routing.latent_dim,
            "quantization_mode": (
                "none"
                if raw_concat or raw_latent
                else (
                    "fixed_jcb_dct_int16"
                    if self.fixed_jcb_packet_codec is not None
                    else ("adaptive_lcf_packet" if self.adaptive_lcf_packet_codec is not None else "fixed_cachejpeg")
                )
            ),
            "quant_table_low": (
                float(getattr(self.fixed_jcb_packet_codec, "low", 0.0))
                if getattr(self, "fixed_jcb_packet_codec", None) is not None
                else None
            ),
            "quant_table_high": (
                float(getattr(self.fixed_jcb_packet_codec, "high", 0.0))
                if getattr(self, "fixed_jcb_packet_codec", None) is not None
                else None
            ),
            "scale_mode": (
                str(getattr(self.fixed_jcb_packet_codec, "scale_mode", "none"))
                if getattr(self, "fixed_jcb_packet_codec", None) is not None
                else "none"
            ),
        }
        transport_total_ms = (
            float(sum(getattr(self.last_transport_stats, name, 0.0) for name in (
                "serialize_seconds", "transmit_seconds", "deserialize_seconds"
            )) * 1000.0)
            if self.last_transport_stats is not None
            else 0.0
        )
        receiver_decode_timing = getattr(self, "_last_receiver_decode_timing", {})
        self.last_codec_stats.update({
            "sharer_prefill_ms": float(sharer_prefill_ms),
            "receiver_prefill_ms": float(receiver_prefill_ms),
            "receiver_first_token_ms": float(receiver_decode_timing.get("receiver_first_token_ms", 0.0)),
            "receiver_decode_ms": float(receiver_decode_timing.get("receiver_decode_ms", 0.0)),
            "transport_total_ms": transport_total_ms,
            "ttft_ms": float(
                sharer_prefill_ms
                + (lcf_encode_seconds + encode_seconds) * 1000.0
                + transport_total_ms
                + (decode_seconds + lcf_decode_seconds) * 1000.0
                + receiver_prefill_ms
                + receiver_decode_timing.get("receiver_first_token_ms", 0.0)
            ),
        })
        if self.adaptive_lcf_packet_codec is not None:
            quantizer_result = self.adaptive_lcf_packet_codec.quantizer.last_result
            if quantizer_result is not None:
                self.last_codec_stats.update(
                    {
                        "estimated_payload_bits": float(
                            quantizer_result.estimated_payload_bits.detach().item()
                        ),
                        "estimated_entropy_bits": float(
                            quantizer_result.estimated_entropy_bits.detach().item()
                        ),
                        "mean_alpha": float(
                            quantizer_result.alpha.detach().float().mean().item()
                        ),
                        "alpha_table_bytes": int(
                            quantizer_result.table_indices.numel()
                        ),
                        "scale_bytes": int(quantizer_result.scale.numel() * 4),
                        "entropy_coder": "zlib",
                    }
                )
        return torch.cat([base_ids, generated], dim=1)

    def _generate_with_concat_layer_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        """Overlap per-layer pre-RoPE concat compression with Sharer prefill."""

        base_ids, teacher_ids, base_mask, teacher_mask = self._split_dual_inputs(
            input_ids, attention_mask
        )
        if self.concat_cache_aligner is None:
            raise RuntimeError("Concat cache aligner was not initialized.")
        teacher_parameter = next(self.teacher_model.parameters())
        routing = self.concat_cache_aligner.prepare_routing()
        self.concat_cache_aligner.prepare_projectors(
            teacher_parameter.device, teacher_parameter.dtype
        )
        zero_sharer_cache = bool(
            self.ablation_config.get("zero_sharer_cache_at_receiver", False)
        )
        streaming = self.eval_codec_config.layer_streaming
        pipeline = ConcatLayerPipeline(
            aligner=self.concat_cache_aligner,
            codec=self.codec,
            codec_config=self.codec_config,
            transport=self.transport,
            routing=routing,
            gpu_streams=streaming.gpu_streams,
            max_inflight_layers=streaming.max_inflight_layers,
            zero_sharer_cache_at_receiver=zero_sharer_cache,
        )
        publisher = StreamingPreRopeKVPublisher(pipeline.submit)
        streaming_cache = StreamingPreRopeDynamicCache(publisher)
        prefill_timer = LayerPrefillTimer(
            self.teacher_model, int(self.teacher_model.config.num_hidden_layers)
        )
        prefill_timer.start()
        finish_attempted = False
        try:
            with stream_pre_rope_keys(self.teacher_model, publisher):
                with torch.no_grad():
                    self.teacher_model(
                        input_ids=teacher_ids,
                        attention_mask=teacher_mask,
                        past_key_values=streaming_cache,
                        use_cache=True,
                    )
            if teacher_parameter.device.type == "cuda":
                torch.cuda.synchronize(teacher_parameter.device)
            layer_prefill_seconds = prefill_timer.finish()
            finish_attempted = True
            receiver_prefix, routing = pipeline.finish()
        except BaseException:
            for handle in prefill_timer.handles:
                handle.remove()
            prefill_timer.handles.clear()
            if not finish_attempted:
                try:
                    pipeline.finish()
                except BaseException:
                    pass
            raise
        pipeline_seconds = pipeline.pipeline_seconds
        self.last_transport_stats = pipeline.aggregate_transport_stats()
        self.last_fusion_stats = self.concat_cache_aligner.last_alignment_stats
        prefix_length = int(receiver_prefix.key_cache[0].shape[2])

        if base_mask is None:
            base_mask = torch.ones_like(base_ids, dtype=torch.long)
        prefix_mask = torch.ones(
            (base_ids.shape[0], prefix_length),
            dtype=base_mask.dtype,
            device=base_mask.device,
        )
        combined_mask = torch.cat([prefix_mask, base_mask], dim=1)
        receiver_position_ids = base_mask.long().cumsum(-1) - 1 + prefix_length
        receiver_position_ids = receiver_position_ids.masked_fill(base_mask == 0, 0)
        with torch.no_grad():
            receiver_outputs = self.base_model(
                input_ids=base_ids,
                attention_mask=combined_mask,
                position_ids=receiver_position_ids,
                past_key_values=receiver_prefix,
                use_cache=True,
            )
        generated = self._decode_from_prefill_outputs(
            receiver_outputs,
            attention_mask=combined_mask,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )

        bandwidth = self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
        ordered_timings = [
            {"layer_idx": layer_idx, **pipeline.layer_timings.get(layer_idx, {})}
            for layer_idx in range(pipeline.num_layers)
        ]
        codec_encode_wall = pipeline.stage_wall_seconds("encode")
        codec_decode_wall = pipeline.stage_wall_seconds("decode")
        lcf_encode_wall = pipeline.stage_wall_seconds("lcf_encode")
        lcf_decode_wall = pipeline.stage_wall_seconds("lcf_decode")
        sender_encode_wall = pipeline.stage_wall_seconds("lcf_encode", "encode")
        receiver_decode_wall = pipeline.stage_wall_seconds("decode", "lcf_decode")
        self.last_codec_stats = {
            "mode": "concat_layer_streaming",
            "num_layers": pipeline.num_layers,
            "gpu_streams": streaming.gpu_streams,
            "max_inflight_layers": streaming.max_inflight_layers,
            "original_kv_bytes": pipeline.original_kv_bytes,
            "lcf_latent_kv_bytes": pipeline.latent_kv_bytes,
            "payload_bytes": pipeline.payload_bytes,
            "compression_factor": (
                float(pipeline.original_kv_bytes / pipeline.payload_bytes)
                if pipeline.payload_bytes
                else 0.0
            ),
            "space_saving_ratio": (
                float(1.0 - pipeline.payload_bytes / pipeline.original_kv_bytes)
                if pipeline.original_kv_bytes
                else 0.0
            ),
            "encode_seconds": codec_encode_wall,
            "decode_seconds": codec_decode_wall,
            "lcf_encode_seconds": lcf_encode_wall,
            "lcf_decode_seconds": lcf_decode_wall,
            "sender_encode_seconds": sender_encode_wall,
            "receiver_decode_seconds": receiver_decode_wall,
            "encode_service_seconds": float(pipeline.codec_encode_seconds),
            "decode_service_seconds": float(pipeline.codec_decode_seconds),
            "lcf_encode_service_seconds": float(pipeline.lcf_encode_seconds),
            "lcf_decode_service_seconds": float(pipeline.lcf_decode_seconds),
            "pipeline_seconds": float(pipeline_seconds),
            "layer_prefill_seconds": [float(value) for value in layer_prefill_seconds],
            "layer_timings": ordered_timings,
            "compute_backend": self.eval_codec_config.codec.compute.backend,
            "transform_dtype": self.eval_codec_config.codec.compute.transform_dtype,
            "entropy_backend": self.eval_codec_config.codec.entropy.backend,
            "layer_streaming": True,
            "layer_execution": "concat_staged_pipeline",
            "transport_mode": self.eval_codec_config.codec.transport.mode,
            "transport_bandwidth_bytes_per_sec": bandwidth,
            "bandwidth_only_transmit_seconds": (
                float(pipeline.payload_bytes / bandwidth) if bandwidth else None
            ),
            "cache_alignment": "concat",
            "concat_projector_type": self.concat_projector_type,
            "codec_order": (
                "lcf_project_kv_cachejpeg_lcf_up"
                if self.concat_projector_type == "lcf_projected_kv"
                else "lcf_down_cachejpeg_lcf_up"
            ),
            "rope_mode": "pre_rope",
            "prefix_tokens": prefix_length,
            "latent_dim": routing.latent_dim,
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
        }
        return torch.cat([base_ids, generated], dim=1)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        if getattr(self, "cache_alignment", "fuser") == "concat":
            if getattr(self, "concat_projector_type", "") == "direct_pre_rope_mlp":
                return self._finalize_timing(self._generate_with_direct_mlp_concat(
                    input_ids, attention_mask, generation_config
                ), route="concat")
            streaming_config = getattr(
                getattr(self, "eval_codec_config", None), "layer_streaming", None
            )
            if streaming_config is not None and streaming_config.enabled:
                return self._finalize_timing(self._generate_with_concat_layer_streaming(
                    input_ids, attention_mask, generation_config
                ), route="concat", execution_mode="layer_streaming")
            return self._finalize_timing(self._generate_with_concat_alignment(
                input_ids, attention_mask, generation_config
            ), route="concat")
        if getattr(self, "fusion_type", "original") == "latent_kv_split":
            return self._finalize_timing(self._generate_with_split_latent(
                input_ids, attention_mask, generation_config
            ), route="fusion")
        streaming_config = getattr(getattr(self, "eval_codec_config", None), "layer_streaming", None)
        if streaming_config is not None and streaming_config.enabled:
            return self._finalize_timing(self._generate_with_layer_streaming(
                input_ids, attention_mask, generation_config
            ), route="fusion", execution_mode="layer_streaming")
        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_dual_inputs(input_ids, attention_mask)
        )
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        sharer_prefill_started = time.perf_counter()
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        sharer_prefill_ms = (time.perf_counter() - sharer_prefill_started) * 1000.0
        receiver_device = next(self.base_model.parameters()).device
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        receiver_prefill_started = time.perf_counter()
        receiver_seed_outputs = self.prefill_on_receiver(
            input_ids=base_input_ids, attention_mask=base_attention_mask
        )
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        receiver_prefill_ms = (time.perf_counter() - receiver_prefill_started) * 1000.0
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_started = time.perf_counter()
        source_codec_enabled = bool(getattr(self, "cache_codec_list", None))
        raw_transport = bool(getattr(self, "raw_cache_transport", False))
        projector_dict = getattr(self.fuser_bridge, "projector_dict", {})
        layer_map = projector_dict.get(
            int(self.assets.base_model_idx), {}
        ).get(int(self.assets.teacher_model_idx), {})
        routed_kv_bytes = 0
        for entry in layer_map.values():
            pairs = entry if isinstance(entry, list) else [entry]
            if not pairs:
                continue
            source_layer, _ = self.fuser_bridge._normalize_pair(pairs[0])
            key, value = legacy_sharer_cache[source_layer]
            routed_kv_bytes += int(
                key.numel() * key.element_size() + value.numel() * value.element_size()
            )
        full_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in legacy_sharer_cache
        )
        original_kv_bytes = full_kv_bytes if raw_transport else (routed_kv_bytes or full_kv_bytes)
        if raw_transport:
            payload = (
                self.raw_dct_prune_codec.encode(legacy_sharer_cache)
                if self.raw_dct_prune_codec is not None
                else tuple(
                    (
                        key.detach().cpu().contiguous(),
                        value.detach().cpu().contiguous(),
                    )
                    for key, value in legacy_sharer_cache
                )
            )
        elif source_codec_enabled:
            payload = self._encode_c2c_source_packet(legacy_sharer_cache)
        else:
            payload = self.encode_cache(legacy_sharer_cache)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_seconds = time.perf_counter() - encode_started
        transport = getattr(self, "transport", None)
        received_payload = transport.roundtrip(payload) if transport is not None else payload
        self.last_transport_stats = transport.last_stats if transport is not None else None
        payload_bytes = int(getattr(self.last_transport_stats, "frame_bytes", 0)) or len(serialize_payload(payload))
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_started = time.perf_counter()
        if raw_transport:
            decoded_teacher_cache = (
                self.raw_dct_prune_codec.decode(received_payload, legacy_sharer_cache)
                if self.raw_dct_prune_codec is not None
                else received_payload
            )
        elif source_codec_enabled:
            decoded_teacher_cache = self._decode_c2c_source_packet(
                received_payload, legacy_sharer_cache
            )
        else:
            decoded_teacher_cache = self.decode_cache(received_payload)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_seconds = time.perf_counter() - decode_started
        zero_sharer_cache = bool(
            getattr(self, "ablation_config", {}).get("zero_sharer_cache_at_receiver", False)
        )
        if zero_sharer_cache:
            decoded_teacher_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in self._to_legacy_cache(decoded_teacher_cache)
            )
        frequency_prune_stats = getattr(payload, "local_summary", {}).get("frequency_prune")
        if frequency_prune_stats is None:
            frequency_prune_stats = self._configured_frequency_prune_stats()
        self.last_codec_stats = {
            "original_kv_bytes": original_kv_bytes,
            "payload_bytes": payload_bytes,
            "compression_factor": float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0,
            "space_saving_ratio": float(1.0 - payload_bytes / original_kv_bytes) if original_kv_bytes else 0.0,
            "encode_seconds": float(encode_seconds),
            "decode_seconds": float(decode_seconds),
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
            "frequency_prune": dict(frequency_prune_stats),
            "raw_dct_prune": (
                dict(self.raw_dct_prune_codec.last_stats or {})
                if self.raw_dct_prune_codec is not None
                else {"enabled": False}
            ),
            "cache_codec": "jcb" if source_codec_enabled else None,
            "quantization_mode": "none" if raw_transport else (
                "fixed_jcb_dct_int16"
                if getattr(self, "fixed_jcb_packet_codec", None) is not None
                else None
            ),
            "quant_table_low": (
                float(getattr(self.fixed_jcb_packet_codec, "low", 0.0))
                if getattr(self, "fixed_jcb_packet_codec", None) is not None
                else None
            ),
            "quant_table_high": (
                float(getattr(self.fixed_jcb_packet_codec, "high", 0.0))
                if getattr(self, "fixed_jcb_packet_codec", None) is not None
                else None
            ),
            "scale_mode": (
                str(getattr(self.fixed_jcb_packet_codec, "scale_mode", "none"))
                if getattr(self, "fixed_jcb_packet_codec", None) is not None
                else "none"
            ),
        }
        fusion_started = time.perf_counter()
        fused_receiver_cache = self.fuse_to_receiver_cache(
            decoded_teacher_cache,
            base_seed_cache=receiver_seed_outputs.past_key_values,
            apply_cache_codec=not source_codec_enabled,
        )
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        transport_total_ms = (
            float(sum(getattr(self.last_transport_stats, name, 0.0) for name in (
                "serialize_seconds", "transmit_seconds", "deserialize_seconds"
            )) * 1000.0)
            if self.last_transport_stats is not None
            else 0.0
        )
        receiver_decode_timing = getattr(self, "_last_receiver_decode_timing", {})
        self.last_codec_stats.update({
            "sharer_prefill_ms": float(sharer_prefill_ms),
            "receiver_prefill_ms": float(receiver_prefill_ms),
            "fusion_ms": float(fusion_ms),
            "receiver_first_token_ms": float(receiver_decode_timing.get("receiver_first_token_ms", 0.0)),
            "receiver_decode_ms": float(receiver_decode_timing.get("receiver_decode_ms", 0.0)),
            "transport_total_ms": transport_total_ms,
            "ttft_ms": float(
                sharer_prefill_ms
                + float(self.last_codec_stats.get("encode_seconds", 0.0)) * 1000.0
                + transport_total_ms
                + float(self.last_codec_stats.get("decode_seconds", 0.0)) * 1000.0
                + fusion_ms
                + receiver_prefill_ms
                + receiver_decode_timing.get("receiver_first_token_ms", 0.0)
            ),
        })
        return self._finalize_timing(torch.cat([base_input_ids, generated], dim=1), route="fusion")


def load_cachejpeg_rosetta_model(
    model_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    cfg = dict(model_config.get("cachejpeg_rosetta_config") or {})
    ablation_config = dict(cfg.get("ablation") or {})
    receiver_only = bool(ablation_config.get("receiver_only", False))
    sharer_only = bool(ablation_config.get("sharer_only", False))
    if receiver_only and sharer_only:
        raise ValueError("receiver_only and sharer_only cannot both be enabled")

    rosetta_config = dict(model_config.get("rosetta_config") or {})
    if receiver_only:
        base_model_name = rosetta_config["base_model"]
        base_model, base_tokenizer = load_hf_model(
            base_model_name,
            device=device,
            generation_config=generation_config,
        )
        # This branch intentionally does not load or execute the sharer/teacher,
        # projector, fuser, CacheJPEG codec, or transport.
        return base_model, base_tokenizer
    if sharer_only:
        teacher_model_name = rosetta_config["teacher_model"]
        teacher_model, teacher_tokenizer = load_hf_model(
            teacher_model_name,
            device=device,
            generation_config=generation_config,
        )
        # This branch intentionally does not load or execute the receiver/base,
        # projector, fuser, CacheJPEG codec, or transport. Returning the teacher
        # tokenizer also ensures the LongBench prompt is tokenized in the
        # sharer's own vocabulary.
        return teacher_model, teacher_tokenizer
    assets = _load_rosetta_assets(model_config, eval_config, device=device, generation_config=generation_config)
    wrapper = CacheJPEGRosettaEvalWrapper(
        assets=assets,
        codec_config=cfg,
    )
    return wrapper, assets.base_tokenizer
