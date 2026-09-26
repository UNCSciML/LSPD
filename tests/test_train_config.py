"""Compose both public training recipes without loading models or starting Ray."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train", ROOT / "scripts/train.py")
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


@pytest.mark.parametrize("method", ["LSPD", "LSPD-RB"])
def test_default_recipes_enable_evaluation_and_persistent_scores(method):
    args, extra = train.parse_args(["--method", method])
    config = train.compose_config(train.build_overrides(args) + extra, method)
    assert config.trainer.val_before_train is True
    assert config.trainer.test_freq == 5
    assert list(config.trainer.logger) == ["console", "file"]
    assert config.trainer.log_eval_by_rollouts is True
    assert config.data.val_files == "data/val.parquet"
    assert config.custom_reward_function.name == "reward_func"
    assert Path(config.custom_reward_function.path).is_file()
    validation = config.actor_rollout_ref.rollout.val_kwargs
    assert validation.n == 16
    assert validation.do_sample is True
    assert validation.temperature == 0.7
    assert validation.top_p == 0.95
    assert validation.max_tokens == 7168
    assert config.actor_rollout_ref.actor.policy_loss.opd_replay_buffer_enabled == (method == "LSPD-RB")
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH == str(
        (Path("outputs") / method.lower() / "metrics.jsonl").resolve()
    )


def test_custom_output_and_evaluation_samples_are_composed(tmp_path):
    output = tmp_path / "experiment with spaces"
    args, extra = train.parse_args(["--output-dir", str(output), "--eval-samples", "4", "--test-freq", "2"])
    config = train.compose_config(train.build_overrides(args) + extra, args.method)
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH == str(output / "metrics.jsonl")
    assert config.actor_rollout_ref.rollout.val_kwargs.n == 4
    assert config.trainer.test_freq == 2


def test_nonpositive_evaluation_sample_count_is_rejected():
    with pytest.raises(SystemExit):
        train.parse_args(["--eval-samples", "0"])
