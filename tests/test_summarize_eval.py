"""Check exact sampled accuracy, persistent logging, and the benchmark score table."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.utils.tracking import FileLogger


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("summarize_eval", ROOT / "scripts/summarize_eval.py")
summarize_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_eval)


def test_empirical_pass_is_distinct_from_average_and_bootstrap():
    # Two prompts: one has exactly one correct answer among 16, the other has none.
    result = process_validation_metrics(
        ["AMC23"] * 32,
        ["question-1"] * 16 + ["question-2"] * 16,
        {"acc": [True] + [False] * 31},
    )["AMC23"]["acc"]
    assert result["mean@16"] == pytest.approx(1 / 32)
    assert result["pass@16"] == pytest.approx(1 / 2)
    assert 0 < result["best@16/mean"] < result["pass@16"]


def test_binary_pass_is_grouped_by_source_and_prompt():
    result = process_validation_metrics(
        ["AMC23", "AIME24", "AIME25"],
        ["shared-id"] * 3,
        {"acc": [True, False, True], "score": [1.0, 0.0, 1.0]},
    )
    assert result["AMC23"]["acc"]["pass@1"] == 1
    assert result["AIME24"]["acc"]["pass@1"] == 0
    assert result["AIME25"]["acc"]["pass@1"] == 1
    assert "pass@1" not in result["AMC23"]["score"]


def test_trainer_validation_logs_all_three_sources_and_exact_pass():
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 16, "do_sample": True}}},
        "reward_model": {"enable": False},
        "data": {"train_batch_size": 64},
        "trainer": {"validation_data_dir": None, "log_eval_by_rollouts": True},
    })
    trainer.global_steps = 0
    trainer.async_rollout_mode = False
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, decode=lambda *args, **kwargs: "text")
    trainer.val_dataloader = [{
        "input_ids": torch.ones((6, 2), dtype=torch.long),
        "data_source": np.array([source for source in ("AMC23", "AIME24", "AIME25") for _ in range(2)], dtype=object),
        "reward_model": np.array([{"ground_truth": "1", "style": "rule"}] * 6, dtype=object),
    }]
    trainer._get_gen_batch = lambda batch: DataProto.from_dict({"input_ids": batch.batch["input_ids"].clone()})
    trainer._maybe_log_val_generations = lambda **kwargs: None

    def generate(batch):
        assert len(batch) == 6 * 16
        return DataProto.from_dict({"responses": torch.ones((len(batch), 1), dtype=torch.long)})

    def reward(batch, return_dict):
        scores = torch.zeros((len(batch), 1))
        scores[::32] = 1
        return {"reward_tensor": scores, "reward_extra_info": {"acc": scores[:, 0].bool().tolist()}}

    trainer.actor_rollout_wg = SimpleNamespace(world_size=1, generate_sequences=generate)
    trainer.val_reward_fn = reward
    metrics = trainer._attach_eval_rollout_axis(trainer._validate())
    for source in ("AMC23", "AIME24", "AIME25"):
        assert metrics[f"val-core/{source}/acc/mean@16"] == pytest.approx(1 / 32)
        assert metrics[f"val-core/{source}/acc/pass@16"] == pytest.approx(1 / 2)
    assert metrics["eval/num_training_rollouts_generated"] == 0
    trainer.global_steps = 5
    assert trainer._attach_eval_rollout_axis({})["eval/num_training_rollouts_generated"] == 1280


def test_baseline_and_training_scores_are_flushed_and_printed(tmp_path, monkeypatch, capsys):
    path = tmp_path / "new-run" / "metrics.jsonl"
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(path))
    logger = FileLogger("LSPD", "LSPD-RB")
    try:
        baseline = {
            f"val-core/{source}/acc/{metric}@16": value
            for source in ("AMC23", "AIME24", "AIME25")
            for metric, value in (("mean", 0.25), ("pass", 0.75))
        }
        logger.log(baseline, step=0)
        assert len(summarize_eval.read_evaluations(path)) == 3  # Visible before close.
        logger.log({"actor/loss": 0.1}, step=1)
        logger.log(baseline, step=5)
        summarize_eval.main([str(path)])
        latest = capsys.readouterr().out
        assert "| 5 | AMC23 | 16 | 25.00 | 75.00 |" in latest
        assert "| 5 | AIME24 |" in latest
        assert "| 5 | AIME25 |" in latest
        assert "| 0 |" not in latest
        summarize_eval.main([str(path), "--all"])
        history = capsys.readouterr().out
        assert "| 0 | AMC23 |" in history
        assert "| 5 | AMC23 |" in history
    finally:
        logger.finish()


def test_legacy_bootstrap_score_is_not_mislabeled_as_pass(tmp_path, capsys):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"step": 0, "data": {
        "val-core/AMC23/acc/mean@16": 0.5,
        "val-core/AMC23/acc/best@16/mean": 0.8,
    }}) + "\n")
    summarize_eval.main([str(path)])
    assert "| 0 | AMC23 | 16 | 50.00 | n/a |" in capsys.readouterr().out


def test_corrupt_or_empty_logs_report_the_problem(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("not json\n")
    with pytest.raises(ValueError, match="invalid metric record"):
        summarize_eval.read_evaluations(path)
    path.write_text(json.dumps({"step": 1, "data": {"actor/loss": 0.1}}) + "\n")
    with pytest.raises(SystemExit):
        summarize_eval.main([str(path)])
