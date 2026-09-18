import torch

from rosetta.baseline.t2t import T2TInference


def test_t2t_tracks_bridge_and_receiver_timing_fields():
    model = T2TInference.__new__(T2TInference)
    model.last_stats = {
        "sharer_decode_ms": 1.0,
        "receiver_encode_ms": 2.0,
        "receiver_prefill_ms": 3.0,
        "receiver_first_token_ms": 4.0,
    }
    assert model.ttft_ms == 10.0


def test_t2t_explicit_ttft_does_not_include_full_receiver_decode_twice():
    model = T2TInference.__new__(T2TInference)
    model.last_stats = {
        "sharer_decode_ms": 10.0,
        "receiver_encode_ms": 2.0,
        "receiver_prefill_ms": 3.0,
        "receiver_first_token_ms": 4.0,
        "receiver_decode_ms": 40.0,
        "ttft_ms": 19.0,
    }
    assert model.ttft_ms == 19.0


def test_t2t_bridge_is_reencoded_by_receiver_tokenizer():
    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return " | ".join(f"{m['role']}:{m['content']}" for m in messages)

        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([[len(text)]], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    model = T2TInference.__new__(T2TInference)
    model.receiver_tokenizer = Tokenizer()
    text = model.build_receiver_prompt("question", "bridge answer")
    assert "assistant:bridge answer" in text
    assert "user:question" in text
