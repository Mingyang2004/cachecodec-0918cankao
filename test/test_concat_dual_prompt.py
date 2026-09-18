import torch

from rosetta.cachejpeg_rosetta.wrapper import CacheJPEGRosettaEvalWrapper
from rosetta.train.dataset_adapters import ConcatDataCollator, ConcatDualPromptDataset


class ToyTokenizer:
    pad_token_id = 0

    def __init__(self, marker: str):
        self.marker = marker

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert not tokenize
        content = "/".join(message["content"] for message in messages)
        suffix = "/assistant" if add_generation_prompt else ""
        return f"{self.marker}:{content}{suffix}"

    def __call__(self, text, *, add_special_tokens: bool):
        assert not add_special_tokens
        return {"input_ids": list(range(1, len(text) + 1))}


def test_concat_dataset_tokenizes_receiver_and_sharer_prompts_independently():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    dataset = ConcatDualPromptDataset(
        [messages],
        receiver_tokenizer=ToyTokenizer("receiver-tokenizer"),
        sharer_tokenizer=ToyTokenizer("sharer"),
        max_length=128,
    )

    sample = dataset[0]

    assert len(sample["receiver_input_ids"]) != len(sample["sharer_prompt_input_ids"])
    assert sample["labels"][: sample["receiver_prompt_length"]] == [
        -100
    ] * sample["receiver_prompt_length"]
    assert sample["labels"][sample["receiver_prompt_length"] :] == sample[
        "receiver_input_ids"
    ][sample["receiver_prompt_length"] :]

    batch = ConcatDataCollator(
        receiver_tokenizer=ToyTokenizer("receiver-tokenizer"),
        sharer_tokenizer=ToyTokenizer("sharer"),
    )([sample])

    assert batch["input_ids"][0].shape[1] == len(sample["receiver_input_ids"])
    assert batch["input_ids"][1].shape[1] == len(sample["sharer_prompt_input_ids"])
    assert torch.equal(batch["attention_mask"][1], torch.ones_like(batch["attention_mask"][1]))
    assert "kv_cache_index" not in batch


def test_cachejpeg_concat_accepts_independent_receiver_and_sharer_inputs():
    receiver_ids = torch.tensor([[1, 2, 3, 4]])
    sharer_ids = torch.tensor([[9, 8]])
    receiver_mask = torch.ones_like(receiver_ids)
    sharer_mask = torch.ones_like(sharer_ids)

    streams = CacheJPEGRosettaEvalWrapper._split_dual_inputs(
        [receiver_ids, sharer_ids], [receiver_mask, sharer_mask]
    )

    assert streams == (receiver_ids, sharer_ids, receiver_mask, sharer_mask)
