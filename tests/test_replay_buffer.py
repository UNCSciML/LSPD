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

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.replay_buffer import DataProtoReplayBuffer


def _rows(row_ids, *, requires_grad=False):
    row_ids = list(row_ids)
    return DataProto.from_dict(
        tensors={
            "row_id": torch.tensor(row_ids),
            "value": torch.tensor(row_ids, dtype=torch.float32, requires_grad=requires_grad).unsqueeze(-1),
        },
        non_tensors={
            "query": np.array([f"query-{row_id}" for row_id in row_ids], dtype=object),
            "payload": np.array([{"response": f"response-{row_id}"} for row_id in row_ids], dtype=object),
        },
        meta_info={"temperature": 0.7},
    )


def test_capacity_and_partial_fifo_eviction():
    buffer = DataProtoReplayBuffer(capacity=5)

    buffer.add(_rows(range(3)))
    buffer.add(_rows(range(3, 7)))

    assert buffer.capacity == 5
    assert len(buffer) == 5
    retained = buffer.sample(5)
    assert sorted(retained.batch["row_id"].tolist()) == [2, 3, 4, 5, 6]


def test_oversized_incoming_batch_replaces_history_with_its_newest_rows():
    buffer = DataProtoReplayBuffer(capacity=4)
    buffer.add(_rows([0, 1, 2]))

    buffer.add(_rows(range(10, 16)))

    assert len(buffer) == 4
    retained = buffer.sample(4)
    assert sorted(retained.batch["row_id"].tolist()) == [12, 13, 14, 15]


def test_add_owns_tensor_and_nested_non_tensor_data_on_cpu():
    source = _rows([1, 2], requires_grad=True)
    buffer = DataProtoReplayBuffer(capacity=2)
    buffer.add(source)

    source.batch["row_id"][0] = 999
    with torch.no_grad():
        source.batch["value"][1] = 999
    source.non_tensor_batch["query"][0] = "changed"
    source.non_tensor_batch["payload"][1]["response"] = "changed"

    stored = buffer.sample(2)
    order = stored.batch["row_id"].argsort()
    stored.reorder(order)
    assert stored.batch["row_id"].tolist() == [1, 2]
    assert stored.batch["value"].squeeze(-1).tolist() == [1.0, 2.0]
    assert stored.batch["value"].device.type == "cpu"
    assert not stored.batch["value"].requires_grad
    assert stored.non_tensor_batch["query"].tolist() == ["query-1", "query-2"]
    assert [item["response"] for item in stored.non_tensor_batch["payload"]] == [
        "response-1",
        "response-2",
    ]
    assert stored.meta_info == {}


def test_sampling_is_reproducible_uniform_and_preserves_row_alignment():
    left = DataProtoReplayBuffer(capacity=10, seed=1234)
    right = DataProtoReplayBuffer(capacity=10, seed=1234)
    for buffer in (left, right):
        buffer.add(_rows(range(4)))
        buffer.add(_rows(range(4, 10)))

    torch.manual_seed(999)
    left_sample = left.sample(6)
    torch.manual_seed(0)
    right_sample = right.sample(6)

    left_ids = left_sample.batch["row_id"].tolist()
    assert left_ids == right_sample.batch["row_id"].tolist()
    assert len(set(left_ids)) == 6
    assert left_sample.non_tensor_batch["query"].tolist() == [f"query-{row_id}" for row_id in left_ids]
    assert [item["response"] for item in left_sample.non_tensor_batch["payload"]] == [
        f"response-{row_id}" for row_id in left_ids
    ]
    assert left_sample.meta_info == {}


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        DataProtoReplayBuffer(capacity=capacity)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_rejects_invalid_sample_size(batch_size):
    buffer = DataProtoReplayBuffer(capacity=2)
    buffer.add(_rows([1]))

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        buffer.sample(batch_size)


def test_rejects_empty_and_undersized_sampling():
    buffer = DataProtoReplayBuffer(capacity=3)
    with pytest.raises(ValueError, match="empty replay buffer"):
        buffer.sample(1)

    buffer.add(_rows([1, 2]))
    with pytest.raises(ValueError, match="cannot sample 3 rows.*containing 2 rows"):
        buffer.sample(3)


def test_empty_add_is_a_noop():
    buffer = DataProtoReplayBuffer(capacity=3)
    buffer.add(_rows([]))
    assert len(buffer) == 0
