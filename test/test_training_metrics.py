import json

import torch

from script.train.SFT_train import build_training_metrics_record, valid_label_token_count


def test_build_training_metrics_record_preserves_losses_and_group_alphas():
    record = build_training_metrics_record(
        phase="adaptive_quant_qat",
        step=12,
        elapsed_seconds=3.5,
        train_loss=1.25,
        task_ce_loss=1.0,
        learning_rate=1e-4,
        grad_norm=0.8,
        epoch=0.25,
        quant_metrics={
            "estimated_payload_bits": 123.0,
            "rate_loss": 0.25,
            "mean_alpha": 0.75,
            "alpha_by_layer_kv": [[0.5, 1.0]],
            "table_indices_by_layer_kv": [[1, 3]],
        },
    )

    assert record["step"] == 12
    assert record["phase"] == "adaptive_quant_qat"
    assert record["total_loss"] == 1.25
    assert record["task_ce_loss"] == 1.0
    assert record["alpha_by_layer_kv"] == [[0.5, 1.0]]
    assert record["table_indices_by_layer_kv"] == [[1, 3]]
    json.dumps(record)


def test_valid_label_token_count_excludes_ignored_prompt_labels():
    labels = torch.tensor([[-100, 4, 5], [-100, -100, 6]])
    assert valid_label_token_count(labels) == 3
