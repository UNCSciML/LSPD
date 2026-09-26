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

"""A bounded, row-oriented replay buffer for :class:`DataProto` batches."""

from __future__ import annotations

import copy
from collections import deque
from numbers import Integral

import torch
from tensordict import TensorDict

from verl import DataProto


def _owned_cpu_copy(data: DataProto) -> DataProto:
    """Copy a ``DataProto`` without retaining device tensors or caller-owned objects."""
    tensor_batch = None
    if data.batch is not None:
        tensor_batch = TensorDict(
            {key: value.detach().to(device="cpu").clone() for key, value in data.batch.items()},
            batch_size=data.batch.batch_size,
            device="cpu",
        )

    return DataProto(
        batch=tensor_batch,
        non_tensor_batch=copy.deepcopy(data.non_tensor_batch),
        meta_info={},
    )


class DataProtoReplayBuffer:
    """Store a bounded FIFO history of ``DataProto`` rows on CPU.

    Incoming batches remain as independent chunks. Consequently, adding a new
    rollout is constant-time except when an old chunk must be trimmed; the
    complete replay history is only assembled for the rows selected by
    :meth:`sample`.
    """

    def __init__(self, capacity: int, seed: int | None = 0):
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")

        self._capacity = int(capacity)
        self._size = 0
        self._chunks: deque[DataProto] = deque()
        self._generator = torch.Generator(device="cpu")
        if seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(seed)

    @property
    def capacity(self) -> int:
        """Maximum number of rows retained by the buffer."""
        return self._capacity

    def __len__(self) -> int:
        return self._size

    def add(self, batch: DataProto) -> None:
        """Append owned CPU copies of ``batch`` rows and evict the oldest rows."""
        if not isinstance(batch, DataProto):
            raise TypeError(f"batch must be a DataProto, got {type(batch).__name__}")

        incoming = len(batch)
        if incoming == 0:
            return

        owned = _owned_cpu_copy(batch)
        if incoming >= self._capacity:
            self._chunks.clear()
            self._chunks.append(owned[incoming - self._capacity : incoming])
            self._size = self._capacity
            return

        self._chunks.append(owned)
        self._size += incoming
        overflow = self._size - self._capacity

        while overflow > 0:
            oldest = self._chunks[0]
            oldest_size = len(oldest)
            if oldest_size <= overflow:
                self._chunks.popleft()
                self._size -= oldest_size
                overflow -= oldest_size
            else:
                self._chunks[0] = oldest[overflow:]
                self._size -= overflow
                overflow = 0

    def sample(self, batch_size: int) -> DataProto:
        """Uniformly sample distinct rows using this buffer's private RNG."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        batch_size = int(batch_size)
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size > self._size:
            raise ValueError(f"cannot sample {batch_size} rows from a replay buffer containing {self._size} rows")

        sampled_indices = torch.randperm(self._size, generator=self._generator)[:batch_size]
        sampled_parts: list[DataProto] = []
        result_positions: list[int] = []
        chunk_start = 0

        for chunk in self._chunks:
            chunk_end = chunk_start + len(chunk)
            in_chunk = (sampled_indices >= chunk_start) & (sampled_indices < chunk_end)
            if in_chunk.any():
                local_indices = sampled_indices[in_chunk] - chunk_start
                sampled_parts.append(chunk.select_idxs(local_indices))
                result_positions.extend(in_chunk.nonzero(as_tuple=False).flatten().tolist())
            chunk_start = chunk_end

        if not sampled_parts:
            raise RuntimeError("replay buffer sampling selected no rows")

        sampled = sampled_parts[0] if len(sampled_parts) == 1 else DataProto.concat(sampled_parts)
        if len(sampled_parts) > 1:
            restore_sample_order = torch.tensor(result_positions).argsort()
            sampled.reorder(restore_sample_order)

        # Advanced tensor/NumPy indexing above owns the arrays themselves. A
        # deep copy also isolates mutable objects stored inside object arrays.
        sampled.non_tensor_batch = copy.deepcopy(sampled.non_tensor_batch)
        sampled.meta_info = {}
        return sampled
