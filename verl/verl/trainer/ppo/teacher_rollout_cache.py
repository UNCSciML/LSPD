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

"""Persistent prompt-level cache for OPD teacher rollouts."""

import hashlib
import json
import os
import sqlite3
import zlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask


class TeacherRolloutCacheMissError(RuntimeError):
    """Raised when strict cache reading encounters an uncached response."""


class TeacherRolloutCache:
    """Store multiple sampled teacher responses for each tokenized prompt.

    Cache keys use non-padding prompt token IDs plus a sample slot. The slot
    distinguishes the ``n`` independently sampled responses required by OPD.
    SQLite access happens only in the trainer process, so Ray workers never
    concurrently write this database.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str, metadata: dict, read_only: bool = False):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.read_only = read_only

        if read_only:
            if not os.path.isfile(self.path):
                raise FileNotFoundError(f"Teacher rollout cache does not exist: {self.path}")
            connection_target = f"file:{Path(self.path).as_posix()}?mode=ro"
            self._connection = sqlite3.connect(connection_target, uri=True, timeout=60)
        else:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._connection = sqlite3.connect(self.path, timeout=60)
            self._create_schema()

        self._connection.execute("PRAGMA busy_timeout = 60000")
        self._validate_metadata(metadata)

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS responses (
                prompt_hash BLOB NOT NULL,
                sample_index INTEGER NOT NULL,
                prompt_tokens BLOB NOT NULL,
                response_tokens BLOB NOT NULL,
                PRIMARY KEY (prompt_hash, sample_index)
            );
            """
        )
        self._connection.commit()

    def _validate_metadata(self, metadata: dict) -> None:
        expected = {
            "schema_version": self.SCHEMA_VERSION,
            **metadata,
        }
        expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))

        try:
            row = self._connection.execute("SELECT value FROM metadata WHERE key = 'cache_config'").fetchone()
        except sqlite3.OperationalError as error:
            raise ValueError(f"Invalid teacher rollout cache schema at {self.path}") from error

        if row is None:
            if self.read_only:
                raise ValueError(f"Teacher rollout cache has no configuration metadata: {self.path}")
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('cache_config', ?)",
                (expected_json,),
            )
            self._connection.commit()
            return

        if row[0] != expected_json:
            actual = json.loads(row[0])
            raise ValueError(
                "Teacher rollout cache configuration mismatch. "
                f"Expected {expected}, found {actual} in {self.path}. Use a different cache path."
            )

    @staticmethod
    def _serialize_tokens(tokens: torch.Tensor | np.ndarray) -> bytes:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        return np.asarray(tokens, dtype="<i4").tobytes()

    @staticmethod
    def _deserialize_response(payload: bytes) -> np.ndarray:
        return np.frombuffer(zlib.decompress(payload), dtype="<i4").astype(np.int64, copy=True)

    @classmethod
    def _prompt_keys(cls, prompts: DataProto, num_samples: int) -> list[tuple[bytes, int, bytes]]:
        input_ids = prompts.batch["input_ids"].detach().cpu()
        attention_mask = prompts.batch["attention_mask"].detach().cpu().bool()
        occurrences: dict[bytes, int] = defaultdict(int)
        keys = []

        for token_row, mask_row in zip(input_ids, attention_mask, strict=True):
            prompt_payload = cls._serialize_tokens(token_row[mask_row])
            prompt_hash = hashlib.sha256(prompt_payload).digest()
            sample_index = occurrences[prompt_hash] % num_samples
            occurrences[prompt_hash] += 1
            keys.append((prompt_hash, sample_index, prompt_payload))

        return keys

    def lookup(self, prompts: DataProto, num_samples: int) -> tuple[torch.Tensor | None, list[str]]:
        """Return cached padded responses, or missing key descriptions."""
        responses = []
        missing = []

        for prompt_hash, sample_index, prompt_payload in self._prompt_keys(prompts, num_samples):
            row = self._connection.execute(
                """
                SELECT prompt_tokens, response_tokens
                FROM responses
                WHERE prompt_hash = ? AND sample_index = ?
                """,
                (prompt_hash, sample_index),
            ).fetchone()
            key_description = f"{prompt_hash.hex()[:16]}:{sample_index}"
            if row is None:
                missing.append(key_description)
                continue
            if row[0] != prompt_payload:
                raise RuntimeError(f"Teacher rollout cache hash collision for key {key_description}")
            responses.append(self._deserialize_response(row[1]))

        if missing:
            return None, missing
        if not responses:
            return torch.empty((0, 0), dtype=torch.long), []

        response_lengths = {response.shape[0] for response in responses}
        if len(response_lengths) != 1:
            raise ValueError(f"Cached teacher responses have inconsistent lengths: {sorted(response_lengths)}")
        return torch.from_numpy(np.stack(responses)), []

    def contains_all(self, prompts: DataProto, num_samples: int) -> bool:
        """Return whether every prompt/sample key exists without decoding responses."""
        for prompt_hash, sample_index, prompt_payload in self._prompt_keys(prompts, num_samples):
            row = self._connection.execute(
                """
                SELECT prompt_tokens
                FROM responses
                WHERE prompt_hash = ? AND sample_index = ?
                """,
                (prompt_hash, sample_index),
            ).fetchone()
            if row is None:
                return False
            if row[0] != prompt_payload:
                key_description = f"{prompt_hash.hex()[:16]}:{sample_index}"
                raise RuntimeError(f"Teacher rollout cache hash collision for key {key_description}")
        return True

    def store(self, prompts: DataProto, responses: torch.Tensor, num_samples: int) -> int:
        """Insert generated responses without replacing already cached samples."""
        if self.read_only:
            raise PermissionError(f"Teacher rollout cache is read-only: {self.path}")
        if len(prompts) != responses.shape[0]:
            raise ValueError(f"Prompt/response batch mismatch: {len(prompts)} != {responses.shape[0]}")

        rows = []
        for (prompt_hash, sample_index, prompt_payload), response in zip(
            self._prompt_keys(prompts, num_samples), responses, strict=True
        ):
            response_payload = zlib.compress(self._serialize_tokens(response), level=1)
            rows.append((prompt_hash, sample_index, prompt_payload, response_payload))

        before = self._connection.total_changes
        with self._connection:
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO responses(prompt_hash, sample_index, prompt_tokens, response_tokens)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        return self._connection.total_changes - before

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        self._connection.close()


def build_teacher_rollouts_from_cached_responses(
    prompts: DataProto,
    responses: torch.Tensor,
    eos_token_id: int | list[int],
    pad_token_id: int,
) -> DataProto:
    """Reconstruct the tensor layout returned by :class:`HFRollout`."""
    prompt_ids = prompts.batch["input_ids"].detach().cpu()
    attention_mask = prompts.batch["attention_mask"].detach().cpu()
    position_ids = prompts.batch["position_ids"].detach().cpu()
    responses = responses.to(dtype=prompt_ids.dtype, device="cpu")

    if prompt_ids.shape[0] != responses.shape[0]:
        raise ValueError(f"Prompt/response batch mismatch: {prompt_ids.shape[0]} != {responses.shape[0]}")

    input_ids = torch.cat((prompt_ids, responses), dim=-1)
    response_length = responses.shape[-1]
    delta_position_id = torch.arange(1, response_length + 1, dtype=position_ids.dtype).unsqueeze(0)
    response_position_ids = position_ids[:, -1:] + delta_position_id
    position_ids = torch.cat((position_ids, response_position_ids), dim=-1)

    response_attention_mask = get_response_mask(
        response_id=responses,
        eos_token=eos_token_id,
        dtype=attention_mask.dtype,
    )
    attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

    return DataProto(
        batch=TensorDict(
            {
                "prompts": prompt_ids,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=prompt_ids.shape[0],
        ),
        non_tensor_batch=deepcopy(prompts.non_tensor_batch),
    )
