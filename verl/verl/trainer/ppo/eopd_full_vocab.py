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

"""Exact full-vocabulary objective helpers for entropy-aware OPD."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


EOPD_STATISTIC_KEYS = (
    "valid_response_token_count",
    "high_entropy_token_count",
    "teacher_entropy_sum",
    "teacher_entropy_selected_sum",
    "full_vocab_forward_kl_selected_sum",
)


def _validate_logit_pair(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> None:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have the same shape for exact "
            f"full-vocabulary forward KL, got {tuple(student_logits.shape)} and "
            f"{tuple(teacher_logits.shape)}."
        )
    if student_logits.ndim < 1 or student_logits.shape[-1] <= 0:
        raise ValueError("Student and teacher logits must have a non-empty vocabulary dimension.")
    if student_logits.device != teacher_logits.device:
        raise ValueError(
            "Student and teacher logits must be on the same device for exact " "full-vocabulary forward KL."
        )


def compute_full_vocab_teacher_entropy_and_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute teacher entropy and exact ``KL(teacher || current student)``.

    Normalization and accumulation use float32. The teacher branch is detached,
    while the student branch remains differentiable.
    """

    _validate_logit_pair(student_logits, teacher_logits)

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()

    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    forward_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return teacher_entropy, forward_kl


def compute_full_vocab_teacher_entropy_and_forward_kl_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    vocab_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the same exact sums in vocabulary chunks to lower peak memory.

    Every vocabulary entry participates and both distributions use a global
    normalizer. Chunking changes only the materialized intermediate size.
    """

    _validate_logit_pair(student_logits, teacher_logits)
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive.")

    student_logits_fp32 = student_logits.float()
    teacher_logits_fp32 = teacher_logits.detach().float()
    student_log_z = torch.logsumexp(student_logits_fp32, dim=-1, keepdim=True)
    teacher_log_z = torch.logsumexp(teacher_logits_fp32, dim=-1, keepdim=True)

    output_shape = teacher_logits_fp32.shape[:-1]
    teacher_entropy = torch.zeros(output_shape, dtype=torch.float32, device=teacher_logits.device)
    forward_kl = torch.zeros(output_shape, dtype=torch.float32, device=student_logits.device)

    vocab_size = teacher_logits_fp32.shape[-1]
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        teacher_log_probs_chunk = teacher_logits_fp32[..., start:end] - teacher_log_z
        student_log_probs_chunk = student_logits_fp32[..., start:end] - student_log_z
        teacher_probs_chunk = teacher_log_probs_chunk.exp()

        teacher_entropy = teacher_entropy - (teacher_probs_chunk * teacher_log_probs_chunk).sum(dim=-1)
        forward_kl = forward_kl + (teacher_probs_chunk * (teacher_log_probs_chunk - student_log_probs_chunk)).sum(
            dim=-1
        )

    return teacher_entropy, forward_kl


def build_response_prediction_mask(
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Map valid response tokens to the causal-logit positions predicting them.

    The first response token is predicted by the final prompt position and the
    final response token is predicted by the position immediately before it.
    """

    if attention_mask.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("attention_mask and response_mask must both be rank-two tensors.")
    if attention_mask.shape[0] != response_mask.shape[0]:
        raise ValueError("attention_mask and response_mask must have the same batch size.")

    response_length = response_mask.shape[-1]
    if attention_mask.shape[-1] < response_length + 1:
        raise ValueError("The input sequence must contain at least one context token before the response.")

    prediction_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    prediction_mask[:, -response_length - 1 : -1] = response_mask.bool()
    return prediction_mask & attention_mask.bool()


def select_valid_response_prediction_logits(
    sequence_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Causally shift dense sequence logits and retain valid response positions."""

    if sequence_logits.ndim != 3 or response_mask.ndim != 2:
        raise ValueError("sequence_logits must be rank three and response_mask must be rank two.")
    if sequence_logits.shape[0] != response_mask.shape[0]:
        raise ValueError("sequence_logits and response_mask must have the same batch size.")

    response_length = response_mask.shape[-1]
    if sequence_logits.shape[1] < response_length + 1:
        raise ValueError("The logit sequence must contain at least one context position before the response.")
    shifted_response_logits = sequence_logits[:, -response_length - 1 : -1, :]
    return shifted_response_logits[response_mask.bool()]


def compute_eopd_full_vocab_objective(
    existing_opd_loss: torch.Tensor,
    student_response_logits: torch.Tensor,
    teacher_response_logits: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_threshold: float,
    forward_kl_coef: float,
    loss_agg_mode: str,
    vocab_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Add gated exact full-vocabulary forward KL to an existing OPD scalar.

    ``student_response_logits`` and ``teacher_response_logits`` contain only
    valid response-token positions, flattened in row-major response order. The
    original response mask is still used for the loss denominator, so the gate
    never changes OPD's reduction semantics.
    """

    if existing_opd_loss.ndim != 0:
        raise ValueError("existing_opd_loss must be a scalar tensor.")
    if response_mask.ndim != 2:
        raise ValueError("response_mask must be a rank-two tensor.")
    if math.isnan(float(entropy_threshold)) or entropy_threshold < 0:
        raise ValueError("entropy_threshold must be non-negative.")
    if not math.isfinite(float(forward_kl_coef)) or forward_kl_coef < 0:
        raise ValueError("forward_kl_coef must be non-negative.")

    valid_mask = response_mask.bool()
    valid_token_count = valid_mask.sum()
    if student_response_logits.ndim != 2 or teacher_response_logits.ndim != 2:
        raise ValueError("Flattened valid response logits must be rank-two tensors.")
    if student_response_logits.shape[0] != int(valid_token_count.item()):
        raise ValueError(
            "The number of response-logit rows must equal response_mask.sum(); "
            f"got {student_response_logits.shape[0]} rows and {int(valid_token_count.item())} valid tokens."
        )

    if vocab_chunk_size is None:
        teacher_entropy, forward_kl = compute_full_vocab_teacher_entropy_and_forward_kl(
            student_response_logits, teacher_response_logits
        )
    else:
        teacher_entropy, forward_kl = compute_full_vocab_teacher_entropy_and_forward_kl_chunked(
            student_response_logits,
            teacher_response_logits,
            vocab_chunk_size=vocab_chunk_size,
        )

    high_entropy_mask = teacher_entropy.detach() > entropy_threshold
    gated_forward_kl = forward_kl * high_entropy_mask.to(dtype=forward_kl.dtype)
    forward_kl_per_token = torch.zeros(
        response_mask.shape,
        dtype=torch.float32,
        device=student_response_logits.device,
    ).masked_scatter(valid_mask, gated_forward_kl)

    # Import locally so the standalone numerical helpers have no dependency on
    # policy-loss registration at module import time.
    from verl.trainer.ppo.core_algos import agg_loss

    forward_kl_loss = agg_loss(
        loss_mat=forward_kl_per_token,
        loss_mask=response_mask.to(dtype=torch.float32),
        loss_agg_mode=loss_agg_mode,
    )
    total_policy_loss = existing_opd_loss + forward_kl_coef * forward_kl_loss

    selected_float = high_entropy_mask.to(dtype=torch.float32)
    statistics = {
        "valid_response_token_count": valid_token_count.detach().to(dtype=torch.float32),
        "high_entropy_token_count": selected_float.sum().detach(),
        "teacher_entropy_sum": teacher_entropy.detach().sum(),
        "teacher_entropy_selected_sum": (teacher_entropy.detach() * selected_float).sum(),
        "full_vocab_forward_kl_selected_sum": (forward_kl.detach() * selected_float).sum(),
    }
    return total_policy_loss, forward_kl_loss, statistics


def build_eopd_metrics(
    opd_loss: torch.Tensor,
    forward_kl_loss: torch.Tensor,
    total_policy_loss: torch.Tensor,
    statistics: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build metrics from already aggregated sums and counts."""

    missing = set(EOPD_STATISTIC_KEYS).difference(statistics)
    if missing:
        raise ValueError(f"Missing EOPD statistics: {sorted(missing)}")

    valid_count = statistics["valid_response_token_count"]
    selected_count = statistics["high_entropy_token_count"]
    zero = torch.zeros((), dtype=torch.float32, device=valid_count.device)

    def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1.0), zero)

    return {
        "eopd/opd_loss": opd_loss.detach(),
        "eopd/full_vocab_forward_kl_loss": forward_kl_loss.detach(),
        "eopd/total_policy_loss": total_policy_loss.detach(),
        "eopd/high_entropy_token_ratio": safe_ratio(selected_count, valid_count),
        "eopd/high_entropy_token_count": selected_count.detach(),
        "eopd/valid_response_token_count": valid_count.detach(),
        "eopd/teacher_entropy_mean": safe_ratio(statistics["teacher_entropy_sum"], valid_count),
        "eopd/teacher_entropy_selected_mean": safe_ratio(statistics["teacher_entropy_selected_sum"], selected_count),
        "eopd/full_vocab_forward_kl_selected_mean": safe_ratio(
            statistics["full_vocab_forward_kl_selected_sum"], selected_count
        ),
    }


def validate_tokenizer_compatibility(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    *,
    teacher_eos_token_id: int | None = None,
    student_eos_token_id: int | None = None,
) -> None:
    """Fail unless teacher and student token IDs denote compatible events.

    A configured EOS remap is the only allowed special-token mismatch. It is
    safe because both tokenizers must still expose the identical token-to-ID
    vocabulary, and the teacher forward explicitly moves the teacher-EOS
    probability mass onto the student's EOS ID before entropy and KL.
    """

    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if student_vocab != teacher_vocab:
        student_size = len(student_vocab)
        teacher_size = len(teacher_vocab)
        raise ValueError(
            "eopd_full_vocab requires identical teacher and student token-to-ID vocabularies; "
            f"got {student_size} student entries and {teacher_size} teacher entries."
        )

    has_teacher_eos = teacher_eos_token_id is not None
    has_student_eos = student_eos_token_id is not None
    if has_teacher_eos != has_student_eos:
        raise ValueError("EOS correction requires both teacher and student EOS token IDs.")

    eos_remap_configured = has_teacher_eos and has_student_eos
    if eos_remap_configured:
        actual_teacher_eos = getattr(teacher_tokenizer, "eos_token_id", None)
        actual_student_eos = getattr(student_tokenizer, "eos_token_id", None)
        if actual_teacher_eos != teacher_eos_token_id or actual_student_eos != student_eos_token_id:
            raise ValueError(
                "Configured EOS correction IDs must match the teacher and student tokenizer EOS assignments; "
                f"configured teacher={teacher_eos_token_id}, student={student_eos_token_id}, "
                f"tokenizers teacher={actual_teacher_eos}, student={actual_student_eos}."
            )

    special_id_attributes = (
        "bos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    )
    if not eos_remap_configured:
        special_id_attributes = (*special_id_attributes, "eos_token_id")
    mismatched_special_ids = {
        attribute: (getattr(student_tokenizer, attribute, None), getattr(teacher_tokenizer, attribute, None))
        for attribute in special_id_attributes
        if getattr(student_tokenizer, attribute, None) != getattr(teacher_tokenizer, attribute, None)
    }
    if mismatched_special_ids:
        raise ValueError(
            "eopd_full_vocab requires compatible teacher and student special-token assignments; "
            f"mismatches: {mismatched_special_ids}."
        )

    if set(getattr(student_tokenizer, "all_special_ids", ())) != set(getattr(teacher_tokenizer, "all_special_ids", ())):
        raise ValueError("eopd_full_vocab requires the same complete set of teacher and student special-token IDs.")
