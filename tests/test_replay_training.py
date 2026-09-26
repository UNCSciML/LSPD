# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""CPU-only coverage for the LSPD-RB replay integration in ``RayPPOTrainer``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.replay_buffer import DataProtoReplayBuffer
from verl.utils.config import validate_opd_replay_config
from verl.workers.config.actor import PolicyLossConfig

REPLAY_TENSOR_KEYS = {
    "responses",
    "response_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "teacher_sample_log_probs",
}


def test_replay_policy_defaults_match_the_off_policy_recipe():
    config = PolicyLossConfig()

    assert config.opd_replay_buffer_max_size == 65536
    assert config.opd_replay_batch_size == 64
    assert config.opd_replay_updates_per_rollout == 256
    assert config.opd_replay_entropy_coeff == pytest.approx(0.1)


def _rollout_batch(batch_size: int = 3) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "responses": torch.arange(batch_size * 2).reshape(batch_size, 2),
            "response_mask": torch.ones(batch_size, 2, dtype=torch.long),
            "input_ids": torch.arange(batch_size * 4).reshape(batch_size, 4),
            "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
            "position_ids": torch.arange(4).repeat(batch_size, 1),
            "teacher_sample_log_probs": torch.full((batch_size, 2), -0.5),
            "advantages": torch.full((batch_size, 2), 99.0),
        },
        non_tensors={"data_source": np.asarray(["test"] * batch_size, dtype=object)},
        meta_info={"temperature": 99.0, "stale_rollout_metadata": True},
    )


class _RecordingAddBuffer:
    capacity = 7

    def __init__(self):
        self.added: list[DataProto] = []

    def add(self, batch: DataProto) -> None:
        self.added.append(batch)

    def __len__(self) -> int:
        return sum(len(batch) for batch in self.added)


def test_add_opd_replay_batch_keeps_only_recomputable_fields_and_logs_size():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.opd_replay_buffer = _RecordingAddBuffer()
    metrics = {"existing": 1.0}

    trainer._add_opd_replay_batch(_rollout_batch(), metrics)

    assert len(trainer.opd_replay_buffer.added) == 1
    stored = trainer.opd_replay_buffer.added[0]
    assert set(stored.batch.keys()) == REPLAY_TENSOR_KEYS
    assert stored.non_tensor_batch == {}
    assert stored.meta_info == {}
    assert metrics == {
        "existing": 1.0,
        "replay/size": 3,
        "replay/max_size": 7,
        "replay/trajectories_added": 3,
    }


class _RecordingSampleBuffer:
    def __init__(self):
        self.sample_sizes: list[int] = []

    def sample(self, batch_size: int) -> DataProto:
        self.sample_sizes.append(batch_size)
        call_index = len(self.sample_sizes) - 1
        masks = (
            torch.tensor([[1, 0, 0], [1, 1, 0]]),
            torch.tensor([[1, 1, 0], [1, 1, 1]]),
            torch.tensor([[1, 1, 1], [1, 0, 0]]),
        )
        return DataProto.from_dict(
            tensors={
                "attention_mask": masks[call_index],
                "sample_id": torch.full((batch_size,), call_index),
            },
            meta_info={"old": "discardable"},
        )


class _RecordingActorWorker:
    def __init__(self):
        self.batches: list[DataProto] = []

    def update_actor(self, batch: DataProto) -> DataProto:
        self.batches.append(batch)
        call_number = len(self.batches)
        return DataProto(
            meta_info={
                "metrics": {
                    "actor/loss": [float(call_number), float(call_number + 2)],
                    "actor/lr": [0.4 - call_number * 0.1],
                    "actor/max_score": [float(call_number), float(call_number + 4)],
                }
            }
        )


def test_update_actor_from_opd_replay_samples_each_update_and_aggregates_metrics():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "opd_replay_batch_size": 2,
                        "opd_replay_updates_per_rollout": 3,
                    }
                },
                "rollout": {"temperature": 0.65, "multi_turn": {"enable": True}},
            },
            "trainer": {"balance_batch": False},
        }
    )
    trainer.opd_replay_buffer = _RecordingSampleBuffer()
    trainer.actor_rollout_wg = _RecordingActorWorker()

    metrics = trainer._update_actor_from_opd_replay()

    assert trainer.opd_replay_buffer.sample_sizes == [2, 2, 2]
    assert len(trainer.actor_rollout_wg.batches) == 3
    assert len({id(batch) for batch in trainer.actor_rollout_wg.batches}) == 3
    assert [batch.meta_info["temperature"] for batch in trainer.actor_rollout_wg.batches] == [0.65] * 3
    assert [batch.meta_info["multi_turn"] for batch in trainer.actor_rollout_wg.batches] == [True] * 3
    assert [batch.meta_info["global_token_num"] for batch in trainer.actor_rollout_wg.batches] == [
        [1, 2],
        [2, 3],
        [3, 1],
    ]
    assert metrics["actor/loss"] == pytest.approx(3.0)
    assert metrics["actor/lr"] == pytest.approx(0.1)
    assert metrics["actor/max_score"] == pytest.approx(7.0)
    assert metrics["replay/batch_size"] == 2
    assert metrics["replay/updates_per_rollout"] == 3
    assert metrics["replay/trajectories_sampled"] == 6


def test_first_four_response_rollout_runs_256_updates_of_64_pairs_without_filling_buffer():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"policy_loss": {}},
                "rollout": {"n": 4, "temperature": 1.0, "multi_turn": {"enable": False}},
            },
            "trainer": {"balance_batch": False},
        }
    )
    trainer.opd_replay_buffer = DataProtoReplayBuffer(capacity=65536, seed=1)
    trainer.actor_rollout_wg = _RecordingActorWorker()
    rollout = _rollout_batch(64).repeat(repeat_times=4, interleave=True)
    # Give each of a prompt's four responses its own stable pair identifier.
    rollout.batch["responses"] = torch.arange(256 * 2).reshape(256, 2)
    metrics = {}

    trainer._add_opd_replay_batch(rollout, metrics)
    metrics.update(trainer._update_actor_from_opd_replay())

    assert len(trainer.opd_replay_buffer) == 256
    assert metrics["replay/trajectories_added"] == 256
    batches = trainer.actor_rollout_wg.batches
    assert len(batches) == 256
    assert all(len(batch) == 64 for batch in batches)
    sampled_ids = [batch.batch["responses"][:, 0].div(2, rounding_mode="floor") for batch in batches]
    assert all(ids.unique().numel() == 64 for ids in sampled_ids)
    all_ids = torch.cat(sampled_ids)
    counts = torch.bincount(all_ids, minlength=256)
    assert all_ids.numel() == 64 * 256
    assert (counts > 1).all()
    assert len({tuple(ids.tolist()) for ids in sampled_ids}) > 1
    # Sampling retains every generated response and leaves the buffer intact.
    stored = trainer.opd_replay_buffer.sample(256)
    assert stored.batch["responses"][:, 0].sort().values.tolist() == list(range(0, 512, 2))
    assert metrics["replay/batch_size"] == 64
    assert metrics["replay/updates_per_rollout"] == 256
    assert metrics["replay/trajectories_sampled"] == 16384


def _valid_replay_config(**policy_loss_overrides):
    policy_loss = {
        "opd_replay_buffer_enabled": True,
        "loss_mode": "opd_squared_logprob",
        **policy_loss_overrides,
    }
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp",
                    "ppo_epochs": 1,
                    "use_kl_loss": False,
                    "policy_loss": policy_loss,
                },
                "rollout": {"n": 4, "log_prob_top_k": 0},
            },
            "reward_model": {"enable": True},
            "data": {"train_batch_size": 64},
            "trainer": {"n_gpus_per_node": 2, "nnodes": 1},
        }
    )


def test_validate_opd_replay_config_accepts_documented_defaults():
    validate_opd_replay_config(_valid_replay_config())


def test_validate_opd_replay_config_is_a_noop_when_replay_is_disabled():
    config = OmegaConf.create({"actor_rollout_ref": {"actor": {"policy_loss": {"opd_replay_buffer_enabled": False}}}})

    validate_opd_replay_config(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"opd_replay_buffer_max_size": 0}, "max size must be positive"),
        ({"opd_replay_batch_size": 0}, "batch size must be positive"),
        (
            {"opd_replay_buffer_max_size": 2, "opd_replay_batch_size": 3},
            "max size must be at least its batch size",
        ),
        ({"opd_replay_updates_per_rollout": 0}, "updates per rollout must be positive"),
        ({"opd_replay_entropy_coeff": -0.1}, "entropy coefficient must be finite and non-negative"),
        ({"opd_replay_entropy_coeff": float("inf")}, "entropy coefficient must be finite and non-negative"),
        ({"opd_replay_seed": 1.5}, "seed must be an integer"),
    ],
)
def test_validate_opd_replay_config_rejects_invalid_replay_parameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_opd_replay_config(_valid_replay_config(**overrides))
