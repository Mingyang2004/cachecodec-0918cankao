"""Token-to-token (T2T) evaluation-only inference baseline."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

from rosetta.cachejpeg.transport import build_transport
from rosetta.utils.evaluate import load_hf_model, set_default_chat_template
from rosetta.utils.timing import merge_present, new_timing_record, transport_fields


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class T2TInference:
    """Generate a textual bridge with the Sharer and re-encode it at Receiver.

    The class intentionally exposes the same ``generate`` shape as a causal LM
    wrapper.  It returns Receiver prompt tokens followed by Receiver answer
    tokens, while ``last_stats`` records the two model stages and TTFT.
    """

    def __init__(
        self,
        receiver_model: Any,
        receiver_tokenizer: Any,
        sharer_model: Any,
        sharer_tokenizer: Any,
        *,
        device: torch.device,
        communication_max_new_tokens: int = 128,
        generation_config: Optional[Dict[str, Any]] = None,
        transport_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.receiver_model = receiver_model
        self.receiver_tokenizer = receiver_tokenizer
        self.sharer_model = sharer_model
        self.sharer_tokenizer = sharer_tokenizer
        self.device = torch.device(device)
        self.communication_max_new_tokens = max(1, int(communication_max_new_tokens))
        self.generation_config = dict(generation_config or {})
        # T2T is a real sender-to-receiver protocol too.  Its link time must
        # therefore use the same transport implementation and prescribed
        # bandwidth as CacheCodec; otherwise the baseline is incomparable.
        self.transport = build_transport(dict(transport_config or {}))
        self.last_stats: Dict[str, Any] | None = None
        self.last_receiver_input_length = 0
        self.last_sharer_input_length = 0
        self.last_bridge_text = ""

    @property
    def last_codec_stats(self):
        """T2T has no cache packet; keep evaluator field unambiguous."""

        return None

    @property
    def ttft_ms(self) -> float:
        if not self.last_stats:
            return 0.0
        return float(
            self.last_stats.get(
                "ttft_ms",
                sum(
                    float(self.last_stats.get(name, 0.0))
                    for name in (
                        "sharer_decode_ms",
                        "receiver_encode_ms",
                        "receiver_prefill_ms",
                        "receiver_first_token_ms",
                    )
                ),
            )
        )

    @property
    def last_transport_stats(self):
        return getattr(self.transport, "last_stats", None)

    @property
    def last_timing_stats(self):
        if not self.last_stats:
            return None
        return self.last_stats.get("timing")

    def build_receiver_prompt(self, prompt: str, bridge_text: str) -> str:
        messages = [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(bridge_text)},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.receiver_tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.receiver_tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _chat_tokenize(tokenizer: Any, prompt: str, device: torch.device):
        messages = [{"role": "user", "content": str(prompt)}]
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            encoded = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict):
            return {key: value.to(device) for key, value in encoded.items()}
        return {"input_ids": encoded.to(device), "attention_mask": torch.ones_like(encoded).to(device)}

    def _generate_tokens(
        self,
        model: Any,
        tokenizer: Any,
        inputs: Dict[str, torch.Tensor],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        model_device = next(model.parameters()).device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)
        _sync(model_device)
        prefill_started = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        _sync(model_device)
        prefill_seconds = time.perf_counter() - prefill_started
        past = outputs.past_key_values
        generated = []
        first_token_started = time.perf_counter()
        logits = outputs.logits[:, -1, :]
        if do_sample:
            scaled = logits if temperature <= 0 else logits / temperature
            next_token = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        _sync(model_device)
        first_token_seconds = time.perf_counter() - first_token_started
        subsequent_decode_seconds = 0.0
        eos_id = getattr(tokenizer, "eos_token_id", None)
        for index in range(max(1, int(max_new_tokens))):
            generated.append(next_token)
            if eos_id is not None and int(next_token[0, 0].item()) == int(eos_id):
                break
            if index + 1 >= max(1, int(max_new_tokens)):
                break
            decode_started = time.perf_counter()
            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            logits = outputs.logits[:, -1, :]
            if do_sample:
                scaled = logits if temperature <= 0 else logits / temperature
                next_token = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            past = outputs.past_key_values
            _sync(model_device)
            # Include subsequent model decode and token selection in decode wall time.
            subsequent_decode_seconds += time.perf_counter() - decode_started
        decode_seconds = max(0.0, first_token_seconds + subsequent_decode_seconds)
        token_ids = torch.cat(generated, dim=1) if generated else input_ids[:, :0]
        return token_ids, {
            "prefill_ms": prefill_seconds * 1000.0,
            "decode_ms": decode_seconds * 1000.0,
            "first_token_ms": first_token_seconds * 1000.0,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        t2t_prompt_text: Optional[str] = None,
        **generation_config: Any,
    ) -> torch.Tensor:
        end_to_end_started = time.perf_counter()
        prompt = t2t_prompt_text
        if prompt is None:
            prompt = self.receiver_tokenizer.decode(
                input_ids[0].detach().cpu().tolist(), skip_special_tokens=True
            )
        do_sample = bool(generation_config.get("do_sample", self.generation_config.get("do_sample", False)))
        temperature = float(generation_config.get("temperature", self.generation_config.get("temperature", 0.0)) or 0.0)
        communication_limit = int(
            generation_config.get(
                "communication_max_new_tokens", self.communication_max_new_tokens
            )
        )
        answer_limit = int(generation_config.get("max_new_tokens", 16))
        sharer_inputs = self._chat_tokenize(self.sharer_tokenizer, prompt, self.device)
        self.last_sharer_input_length = int(sharer_inputs["input_ids"].shape[1])
        sharer_tokens, sharer_timing = self._generate_tokens(
            self.sharer_model,
            self.sharer_tokenizer,
            sharer_inputs,
            max_new_tokens=communication_limit,
            do_sample=do_sample,
            temperature=temperature,
        )
        bridge_text = self.sharer_tokenizer.decode(
            sharer_tokens[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        self.last_bridge_text = bridge_text

        # Wire the generated bridge through exactly one framed transport.  The
        # byte count includes serialization and the framing header, not Python
        # character count (UTF-8 can take multiple bytes per character).
        bridge_payload = {"version": 1, "text": bridge_text}
        received_bridge = self.transport.roundtrip(bridge_payload)
        bridge_text = str(received_bridge["text"])

        receiver_prompt = self.build_receiver_prompt(prompt, bridge_text)
        receiver_tokenize_started = time.perf_counter()
        receiver_tokens = self.receiver_tokenizer(
            receiver_prompt, return_tensors="pt", add_special_tokens=False
        )
        receiver_tokenize_ms = (time.perf_counter() - receiver_tokenize_started) * 1000.0
        receiver_h2d_started = time.perf_counter()
        receiver_inputs = {
            key: value.to(self.device) for key, value in receiver_tokens.items()
            if isinstance(value, torch.Tensor)
        }
        _sync(self.device)
        receiver_h2d_ms = (time.perf_counter() - receiver_h2d_started) * 1000.0
        receiver_encode_ms = receiver_tokenize_ms + receiver_h2d_ms
        self.last_receiver_input_length = int(receiver_inputs["input_ids"].shape[1])
        answer_tokens, receiver_timing = self._generate_tokens(
            self.receiver_model,
            self.receiver_tokenizer,
            receiver_inputs,
            max_new_tokens=answer_limit,
            do_sample=do_sample,
            temperature=temperature,
        )
        timing = new_timing_record(route="t2t", bridge="text", codec="none")
        merge_present(timing, transport_fields(self.last_transport_stats))
        merge_present(timing, {
            "sharer_prefill_ms": sharer_timing["prefill_ms"],
            "sharer_decode_ms": sharer_timing["decode_ms"],
            "receiver_reencode_ms": receiver_encode_ms,
            "receiver_prefill_ms": receiver_timing["prefill_ms"],
            "receiver_first_token_ms": receiver_timing["first_token_ms"],
            "receiver_decode_ms": receiver_timing["decode_ms"],
        })
        timing["ttft_ms"] = (
            sharer_timing["prefill_ms"] + sharer_timing["decode_ms"]
            + float(timing["transport_wall_ms"] or 0.0)
            + receiver_encode_ms + receiver_timing["prefill_ms"]
            + receiver_timing["first_token_ms"]
        )
        timing["model_e2e_internal_ms"] = (time.perf_counter() - end_to_end_started) * 1000.0
        self.last_stats = {
            "mode": "t2t",
            "sharer_input_tokens": self.last_sharer_input_length,
            "sharer_decode_tokens": int(sharer_tokens.shape[1]),
            "bridge_text_length": len(bridge_text),
            "bridge_utf8_bytes": len(bridge_text.encode("utf-8")),
            "receiver_reencoded_tokens": self.last_receiver_input_length,
            "sharer_prefill_ms": sharer_timing["prefill_ms"],
            "sharer_token_decode_ms": sharer_timing["decode_ms"],
            "sharer_decode_ms": sharer_timing["prefill_ms"] + sharer_timing["decode_ms"],
            "receiver_encode_ms": receiver_encode_ms,
            "receiver_tokenize_ms": receiver_tokenize_ms,
            "receiver_h2d_ms": receiver_h2d_ms,
            "receiver_prefill_ms": receiver_timing["prefill_ms"],
            "receiver_token_decode_ms": receiver_timing["decode_ms"],
            "receiver_first_token_ms": receiver_timing["first_token_ms"],
            "receiver_decode_ms": receiver_timing["decode_ms"],
            "ttft_ms": (
                sharer_timing["prefill_ms"]
                + sharer_timing["decode_ms"]
                + float(timing["transport_wall_ms"] or 0.0)
                + receiver_encode_ms
                + receiver_timing["prefill_ms"]
                + receiver_timing["first_token_ms"]
            ),
            "end_to_end_components_ms": {
                "sharer_decode": sharer_timing["prefill_ms"] + sharer_timing["decode_ms"],
                "receiver_encode": receiver_encode_ms,
                "receiver_prefill": receiver_timing["prefill_ms"],
                "receiver_decode": receiver_timing["decode_ms"],
            },
            "internal_end_to_end_latency_ms": timing["model_e2e_internal_ms"],
            "timing": timing,
        }
        return torch.cat([receiver_inputs["input_ids"], answer_tokens], dim=1)


def load_t2t_model(
    model_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[T2TInference, Any]:
    cfg = dict(model_config.get("t2t_config") or {})
    receiver_name = (
        cfg.get("receiver_model")
        or model_config.get("receiver_model")
        or cfg.get("answer_model")
    )
    sharer_name = (
        cfg.get("sharer_model")
        or model_config.get("sharer_model")
        or cfg.get("context_model")
    )
    if not receiver_name or not sharer_name:
        raise ValueError(
            "T2T requires t2t_config.receiver_model and t2t_config.sharer_model."
        )
    receiver_model, receiver_tokenizer = load_hf_model(
        str(receiver_name), device=device, generation_config=generation_config
    )
    sharer_model, sharer_tokenizer = load_hf_model(
        str(sharer_name), device=device, generation_config=generation_config
    )
    if receiver_tokenizer.pad_token is None:
        receiver_tokenizer.pad_token = receiver_tokenizer.eos_token
    if sharer_tokenizer.pad_token is None:
        sharer_tokenizer.pad_token = sharer_tokenizer.eos_token
    set_default_chat_template(receiver_tokenizer, str(receiver_name))
    set_default_chat_template(sharer_tokenizer, str(sharer_name))
    model = T2TInference(
        receiver_model=receiver_model,
        receiver_tokenizer=receiver_tokenizer,
        sharer_model=sharer_model,
        sharer_tokenizer=sharer_tokenizer,
        device=device,
        communication_max_new_tokens=int(cfg.get("communication_max_new_tokens", 128)),
        generation_config=generation_config,
        transport_config=dict(cfg.get("transport") or model_config.get("transport") or {}),
    )
    return model, receiver_tokenizer


__all__ = ["T2TInference", "load_t2t_model"]
