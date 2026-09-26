# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = [
    "omega_conf_to_dataclass",
    "validate_config",
    "validate_eopd_full_vocab_config",
    "validate_opd_replay_config",
]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_eopd_full_vocab_config(config: DictConfig) -> None:
    """Validate requirements specific to exact full-vocabulary entropy-aware OPD."""
    actor_config = config.actor_rollout_ref.actor
    policy_loss_config = actor_config.policy_loss
    if policy_loss_config.get("loss_mode", "vanilla") != "eopd_full_vocab":
        return

    entropy_threshold = policy_loss_config.get("eopd_entropy_threshold", 0.8)
    if math.isnan(float(entropy_threshold)) or entropy_threshold < 0:
        raise ValueError("eopd_full_vocab requires policy_loss.eopd_entropy_threshold >= 0")

    forward_kl_coef = policy_loss_config.get("eopd_forward_kl_coef", 1.0)
    if not math.isfinite(float(forward_kl_coef)) or forward_kl_coef < 0:
        raise ValueError("eopd_full_vocab requires policy_loss.eopd_forward_kl_coef >= 0")

    chunk_size = policy_loss_config.get("eopd_full_vocab_chunk_size", 4096)
    if chunk_size <= 0:
        raise ValueError("eopd_full_vocab requires policy_loss.eopd_full_vocab_chunk_size > 0")

    rollout_config = config.actor_rollout_ref.rollout
    if rollout_config.n != 1:
        raise ValueError(
            "eopd_full_vocab requires exactly one student rollout per prompt: "
            "actor_rollout_ref.rollout.n must equal 1"
        )
    if rollout_config.get("log_prob_top_k", 0) != 0:
        raise ValueError(
            "eopd_full_vocab requires actor_rollout_ref.rollout.log_prob_top_k == 0; "
            "the method does not use top-k candidates"
        )

    if config.algorithm.adv_estimator != "token_reward_direct":
        raise ValueError("eopd_full_vocab requires algorithm.adv_estimator=token_reward_direct")

    reward_model_config = config.reward_model
    if not reward_model_config.enable:
        raise ValueError("eopd_full_vocab requires reward_model.enable=True for teacher scoring")
    if reward_model_config.enable_resource_pool:
        raise ValueError("eopd_full_vocab requires reward_model.enable_resource_pool=False")

    supported_strategies = {"fsdp", "fsdp2"}
    if actor_config.strategy not in supported_strategies:
        raise ValueError("eopd_full_vocab requires actor strategy fsdp or fsdp2")
    if reward_model_config.strategy not in supported_strategies:
        raise ValueError("eopd_full_vocab requires reward-model strategy fsdp or fsdp2")

    if actor_config.get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("eopd_full_vocab requires actor Ulysses sequence parallel size == 1")
    if reward_model_config.get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("eopd_full_vocab requires reward-model Ulysses sequence parallel size == 1")

    if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
        raise ValueError(
            "eopd_full_vocab currently requires the legacy FSDP actor/reward workers; "
            "trainer.use_legacy_worker_impl must be auto or enable"
        )

    teacher_eos_token_id = rollout_config.get("opd_teacher_eos_token_id", None)
    student_eos_token_id = rollout_config.get("opd_student_eos_token_id", None)
    if (teacher_eos_token_id is None) != (student_eos_token_id is None):
        raise ValueError(
            "eopd_full_vocab EOS correction requires both opd_teacher_eos_token_id "
            "and opd_student_eos_token_id"
        )
    if teacher_eos_token_id is not None:
        if not isinstance(teacher_eos_token_id, int) or teacher_eos_token_id < 0:
            raise ValueError("eopd_full_vocab requires a non-negative integer teacher EOS token ID")
        if not isinstance(student_eos_token_id, int) or student_eos_token_id < 0:
            raise ValueError("eopd_full_vocab requires a non-negative integer student EOS token ID")

    if actor_config.entropy_coeff != 0:
        raise ValueError("eopd_full_vocab requires actor entropy_coeff == 0; teacher entropy is only a gate")
    if actor_config.use_kl_loss:
        raise ValueError("eopd_full_vocab requires actor use_kl_loss=False")


def validate_opd_replay_config(config: DictConfig) -> None:
    """Validate the replay-backed sampled squared-log-prob OPD recipe."""
    actor_config = config.actor_rollout_ref.actor
    policy_loss_config = actor_config.policy_loss
    if not policy_loss_config.get("opd_replay_buffer_enabled", False):
        return

    if policy_loss_config.get("loss_mode", "vanilla") != "opd_squared_logprob":
        raise ValueError(
            "OPD replay requires actor.policy_loss.loss_mode=opd_squared_logprob"
        )
    if actor_config.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("OPD replay currently requires actor strategy fsdp or fsdp2")
    if actor_config.ppo_epochs != 1:
        raise ValueError("OPD replay requires actor.ppo_epochs=1")
    if actor_config.use_kl_loss:
        raise ValueError("OPD replay requires actor.use_kl_loss=False")

    rollout_config = config.actor_rollout_ref.rollout
    if rollout_config.get("log_prob_top_k", 0) != 0:
        raise ValueError(
            "OPD replay requires actor_rollout_ref.rollout.log_prob_top_k=0"
        )
    if not config.reward_model.enable:
        raise ValueError("OPD replay requires reward_model.enable=True for teacher targets")
    if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
        raise ValueError(
            "OPD replay currently requires the legacy reward-model worker; "
            "trainer.use_legacy_worker_impl must be auto or enable"
        )

    max_size = int(policy_loss_config.get("opd_replay_buffer_max_size", 65536))
    batch_size = int(policy_loss_config.get("opd_replay_batch_size", 64))
    num_updates = int(policy_loss_config.get("opd_replay_updates_per_rollout", 256))
    entropy_coeff = float(policy_loss_config.get("opd_replay_entropy_coeff", 0.1))
    replay_seed = policy_loss_config.get("opd_replay_seed", 1)
    if max_size <= 0:
        raise ValueError("OPD replay buffer max size must be positive")
    if batch_size <= 0:
        raise ValueError("OPD replay batch size must be positive")
    if max_size < batch_size:
        raise ValueError("OPD replay buffer max size must be at least its batch size")
    if num_updates <= 0:
        raise ValueError("OPD replay updates per rollout must be positive")
    if not math.isfinite(entropy_coeff) or entropy_coeff < 0:
        raise ValueError("OPD replay entropy coefficient must be finite and non-negative")
    if not isinstance(replay_seed, int):
        raise ValueError("OPD replay seed must be an integer")

    rollout_batch_size = int(config.data.train_batch_size) * int(rollout_config.n)
    if rollout_batch_size < batch_size:
        raise ValueError(
            "OPD replay needs each rollout batch to contain at least one replay batch: "
            f"got {rollout_batch_size} trajectories for replay batch size {batch_size}"
        )

    n_actor_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    sequence_parallel_size = int(actor_config.get("ulysses_sequence_parallel_size", 1))
    if n_actor_gpus % sequence_parallel_size != 0:
        raise ValueError("Actor GPU count must be divisible by Ulysses sequence parallel size")
    data_parallel_size = n_actor_gpus // sequence_parallel_size
    if batch_size % data_parallel_size != 0:
        raise ValueError(
            "OPD replay batch size must be divisible by the actor data-parallel size: "
            f"got batch size {batch_size} and data-parallel size {data_parallel_size}"
        )


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    validate_eopd_full_vocab_config(config)
    validate_opd_replay_config(config)

    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    # Check for reward model micro-batch size conflicts
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )

    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        assert config.actor_rollout_ref.model.lora_rank <= 512, "LoRA rank in vLLM must be less than or equal to 512"

    print("[validate_config] All configuration checks passed successfully!")
