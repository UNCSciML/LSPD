#!/usr/bin/env python3
"""Portable launch entry point for LSPD and LSPD-RB."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("LSPD", "LSPD-RB"), default="LSPD")
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-data", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--val-data", type=Path, default=Path("data/val.parquet"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--actor-gpus", type=positive_int, default=2)
    parser.add_argument("--teacher-gpus", type=positive_int, default=2)
    parser.add_argument("--cpus", type=positive_int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--train-batch-size", type=positive_int, default=64)
    parser.add_argument("--mini-batch-size", type=positive_int, default=16)
    parser.add_argument("--responses", type=positive_int, default=4)
    parser.add_argument("--max-prompt-length", type=positive_int, default=1024)
    parser.add_argument("--max-response-length", type=positive_int, default=7168)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--entropy-coeff", type=float, default=0.1)
    parser.add_argument("--penalty", choices=("huber", "squared"), default="huber")
    parser.add_argument("--log-ratio-clip", type=float, default=5.0)
    parser.add_argument("--replay-capacity", type=positive_int, default=65536)
    parser.add_argument("--replay-batch-size", type=positive_int, default=64)
    parser.add_argument("--replay-updates", type=positive_int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--test-freq", type=int, default=5, help="Evaluate every N outer training steps (default: 5); -1 disables periodic evaluation.")
    parser.add_argument("--eval-samples", type=positive_int, default=16, help="Sampled answers per evaluation problem (default: 16).")
    parser.add_argument("--dry-run", action="store_true", help="Print the command; no dependencies, models or GPUs required.")
    parser.add_argument("--check-config", action="store_true", help="Compose and print the configuration without loading models or starting Ray.")
    args, overrides = parser.parse_known_args(argv)
    if not all(math.isfinite(value) for value in (args.entropy_coeff, args.learning_rate, args.log_ratio_clip)):
        parser.error("entropy coefficient, learning rate and log-ratio threshold must be finite")
    if args.entropy_coeff < 0 or args.learning_rate <= 0 or args.log_ratio_clip <= 0:
        parser.error("entropy coefficient must be nonnegative; learning rate and log-ratio threshold must be positive")
    if args.replay_capacity < args.replay_batch_size:
        parser.error("replay capacity must be at least replay batch size")
    if args.method == "LSPD" and args.train_batch_size % args.mini_batch_size:
        parser.error("train batch size must be divisible by mini-batch size")
    if args.method == "LSPD-RB" and args.replay_batch_size % args.actor_gpus:
        parser.error("replay batch size must be divisible by actor GPUs")
    if any("=" not in item or item.startswith("--") for item in overrides):
        parser.error("extra arguments must be Hydra key=value overrides")
    return args, overrides


def quoted(value: str | Path) -> str:
    return json.dumps(str(value))


def build_overrides(args: argparse.Namespace) -> list[str]:
    replay = args.method == "LSPD-RB"
    output = args.output_dir or Path("outputs") / args.method.lower()
    mini_batch = args.replay_batch_size if replay else args.mini_batch_size
    length = args.max_prompt_length + args.max_response_length
    values = {
        "algorithm.adv_estimator": "token_reward_direct",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": quoted(args.train_data),
        "data.val_files": quoted(args.val_data),
        "data.shuffle": False,
        "data.train_batch_size": args.train_batch_size,
        "data.dataloader_num_workers": 0,
        "data.max_prompt_length": args.max_prompt_length,
        "data.max_response_length": args.max_response_length,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.return_raw_chat": True,
        "+data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.model.path": quoted(args.student),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_activation_offload": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": args.learning_rate,
        "actor_rollout_ref.actor.ppo_mini_batch_size": mini_batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": length,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "opd_squared_logprob",
        "actor_rollout_ref.actor.policy_loss.opd_entropy_coeff": args.entropy_coeff,
        "actor_rollout_ref.actor.policy_loss.opd_log_ratio_clip": args.log_ratio_clip,
        "actor_rollout_ref.actor.policy_loss.opd_topk_weight_mode": "student_p",
        "actor_rollout_ref.actor.policy_loss.opd_penalty_mode": args.penalty,
        "actor_rollout_ref.actor.policy_loss.opd_replay_buffer_enabled": replay,
        "actor_rollout_ref.actor.policy_loss.opd_replay_buffer_max_size": args.replay_capacity,
        "actor_rollout_ref.actor.policy_loss.opd_replay_batch_size": args.replay_batch_size,
        "actor_rollout_ref.actor.policy_loss.opd_replay_updates_per_rollout": args.replay_updates,
        "actor_rollout_ref.actor.policy_loss.opd_replay_seed": args.seed,
        "actor_rollout_ref.actor.policy_loss.opd_replay_entropy_coeff": args.entropy_coeff,
        "actor_rollout_ref.actor.fsdp_config.param_offload": not replay,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": not replay,
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch": True,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.repetition_penalty": 1.0,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.8,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 32768,
        "actor_rollout_ref.rollout.max_model_len": length,
        "actor_rollout_ref.rollout.n": args.responses,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.calculate_log_probs": False,
        "+actor_rollout_ref.rollout.log_prob_top_k": 0,
        "+actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "+actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "+actor_rollout_ref.rollout.teacher_temperature": 1.0,
        "+actor_rollout_ref.rollout.stop_token_ids": "[151645,151643]",
        "+actor_rollout_ref.rollout.opd_teacher_eos_token_id": 151645,
        "+actor_rollout_ref.rollout.opd_student_eos_token_id": 151643,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.n": args.eval_samples,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.7,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 0.95,
        "+actor_rollout_ref.rollout.val_kwargs.max_tokens": args.max_response_length,
        "reward_model.enable": True,
        "reward_model.enable_resource_pool": True,
        "reward_model.n_gpus_per_node": args.teacher_gpus,
        "reward_model.nnodes": 1,
        "reward_model.model.path": quoted(args.teacher),
        "reward_model.model.input_tokenizer": "null",
        "reward_model.model.use_remove_padding": True,
        "reward_model.model.use_fused_kernels": False,
        "reward_model.model.fsdp_config.param_offload": False,
        "+reward_model.model.dtype": "fp32",
        "reward_model.micro_batch_size_per_gpu": 24,
        "+reward_model.reward_kwargs.enable_format_reward": False,
        "custom_reward_function.path": quoted(ROOT / "verl/verl/utils/reward_score/ttrl_math/__init__.py"),
        "custom_reward_function.name": "reward_func",
        "trainer.val_before_train": True,
        "trainer.log_val_generations": 0,
        "trainer.logger": "[console,file]",
        "trainer.project_name": "LSPD",
        "trainer.experiment_name": args.method,
        "trainer.rollout_data_dir": "null",
        "trainer.validation_data_dir": "null",
        "trainer.n_gpus_per_node": args.actor_gpus,
        "trainer.nnodes": 1,
        "trainer.save_freq": args.save_freq,
        "trainer.test_freq": args.test_freq,
        "trainer.total_training_steps": args.steps,
        "+trainer.log_eval_by_rollouts": True,
        "trainer.total_epochs": 1,
        "trainer.default_local_dir": quoted(output),
        "trainer.resume_mode": "disable",
        "trainer.resume_from_path": "null",
        "trainer.is_plot": False,
        "ray_kwargs.ray_init.num_cpus": args.cpus,
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH": quoted(output.resolve() / "metrics.jsonl"),
        "hydra.run.dir": quoted(output / "hydra"),
        "hydra.job.chdir": False,
    }
    return [f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in values.items()]


def compose_config(overrides: list[str], method: str):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = ROOT / "verl/verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(config)
    required = {
        "algorithm.adv_estimator": "token_reward_direct",
        "algorithm.use_kl_in_reward": False,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "opd_squared_logprob",
        "actor_rollout_ref.actor.policy_loss.opd_replay_buffer_enabled": method == "LSPD-RB",
        "actor_rollout_ref.actor.policy_loss.opd_teacher_rollout_cache_mode": "off",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.log_prob_top_k": 0,
        "actor_rollout_ref.rollout.opd_teacher_eos_token_id": 151645,
        "actor_rollout_ref.rollout.opd_student_eos_token_id": 151643,
        "reward_model.enable": True,
        "reward_model.model.use_fused_kernels": False,
        "data.apply_chat_template_kwargs.enable_thinking": False,
    }
    for key, expected in required.items():
        if OmegaConf.select(config, key) != expected:
            raise ValueError(f"{method} requires {key}={expected!r}")
    if set(config.actor_rollout_ref.rollout.stop_token_ids) != {151645, 151643}:
        raise ValueError("The rollout must stop on both student and teacher EOS tokens")
    loss = config.actor_rollout_ref.actor.policy_loss
    if loss.opd_penalty_mode not in {"huber", "squared"}:
        raise ValueError("LSPD supports huber or squared penalties")
    for key in ("opd_entropy_coeff", "opd_replay_entropy_coeff", "opd_log_ratio_clip"):
        value = float(loss[key])
        if not math.isfinite(value) or value < 0 or (key == "opd_log_ratio_clip" and value == 0):
            raise ValueError(f"Invalid loss coefficient: {key}={value}")
    return config


def main() -> None:
    args, extra = parse_args()
    overrides = build_overrides(args) + extra
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "verl"), str(ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("NCCL_DEBUG", "WARN")
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    if args.dry_run:
        print("PYTHONPATH=" + shlex.quote(env["PYTHONPATH"]) + " " + shlex.join(command))
        return
    config = compose_config(overrides, args.method)
    if args.check_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(config))
        return
    for field in (config.data.train_files, config.data.val_files):
        if not isinstance(field, str):
            raise SystemExit("Use one Parquet file for each of data.train_files and data.val_files.")
        path = Path(field)
        if not path.is_file():
            raise SystemExit(f"Missing data file: {path}. Run scripts/prepare_data.py for training data and scripts/prepare_benchmarks.py for evaluation data.")
    sys.path.insert(0, str(ROOT))
    from lspd.preflight import check_model_pair

    check_model_pair(config.actor_rollout_ref.model.path, config.reward_model.model.path)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
