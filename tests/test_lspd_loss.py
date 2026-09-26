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

"""LSPD sampled-token objective, entropy, masking, and gradient checks."""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_opd_squared_logprob_objective


def test_response_means_and_entropy_use_only_valid_tokens():
    student_logp = torch.tensor([[-1.0, -2.0, -9.0], [-3.0, -1.0, -2.0]], requires_grad=True)
    teacher_logp = torch.tensor([[-2.0, -4.0, 100.0], [-1.0, -2.0, -3.0]], requires_grad=True)
    entropy = torch.tensor([[2.0, 3.0, 100.0], [1.0, 2.0, 3.0]], requires_grad=True)
    objective, metrics = compute_opd_squared_logprob_objective(
        log_prob=student_logp,
        teacher_log_prob=teacher_logp,
        entropy=entropy,
        response_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        entropy_coeff=0.1,
        penalty_mode="huber",
        log_ratio_clip=5.0,
    )

    # Mean squared errors per response: 2.5 and 2.0; entropy means: 2.5 and 2.0.
    assert objective.item() == pytest.approx(2.25 - 0.1 * 2.25)
    assert metrics["actor/opd_sampled_full_vocab"] == 1.0
    assert metrics["actor/opd_topk_effective_k"] == 1.0
    objective.backward()
    assert teacher_logp.grad is None
    assert student_logp.grad[0, 2].item() == 0.0
    assert entropy.grad[0, 2].item() == 0.0
    assert entropy.grad[0, 0].item() == pytest.approx(-0.025)


@pytest.mark.parametrize("gap,penalty,gradient", [(2.0, 4.0, 4.0), (10.0, 75.0, 10.0), (-10.0, 75.0, -10.0)])
def test_huber_threshold_five_has_nonzero_bounded_tail_gradient(gap, penalty, gradient):
    student_logp = torch.tensor([[gap]], requires_grad=True)
    objective, _ = compute_opd_squared_logprob_objective(
        log_prob=student_logp,
        teacher_log_prob=torch.zeros(1, 1),
        entropy=torch.tensor([[3.0]]),
        response_mask=torch.ones(1, 1),
        entropy_coeff=0.1,
        penalty_mode="huber",
        log_ratio_clip=5.0,
    )
    assert objective.item() == pytest.approx(penalty - 0.3)
    objective.backward()
    assert student_logp.grad.item() == pytest.approx(gradient)


def test_sampled_actions_receive_no_extra_probability_weight():
    results = []
    for weight_mode in ("uniform", "student_p"):
        loss, _ = compute_opd_squared_logprob_objective(
            log_prob=torch.tensor([[-1.0, -2.0]]),
            teacher_log_prob=torch.tensor([[-2.0, -4.0]]),
            entropy=torch.zeros(1, 2),
            response_mask=torch.ones(1, 2),
            topk_weight_mode=weight_mode,
            penalty_mode="huber",
            log_ratio_clip=5.0,
        )
        results.append(loss.item())
    assert results == pytest.approx([2.5, 2.5])


def test_repeating_response_tokens_preserves_sequence_mean():
    results = []
    for length in (1, 4):
        loss, _ = compute_opd_squared_logprob_objective(
            log_prob=torch.full((1, length), -1.0),
            teacher_log_prob=torch.full((1, length), -3.0),
            entropy=torch.full((1, length), 2.0),
            response_mask=torch.ones(1, length),
            entropy_coeff=0.1,
            penalty_mode="huber",
            log_ratio_clip=5.0,
        )
        results.append(loss.item())
    assert results == pytest.approx([3.8, 3.8])


def test_plain_squared_option_preserves_unbounded_tail_gradient():
    student_logp = torch.tensor([[10.0]], requires_grad=True)
    objective, metrics = compute_opd_squared_logprob_objective(
        log_prob=student_logp,
        teacher_log_prob=torch.zeros(1, 1),
        entropy=torch.zeros(1, 1),
        response_mask=torch.ones(1, 1),
        penalty_mode="squared",
        log_ratio_clip=5.0,
    )
    assert objective.item() == pytest.approx(100.0)
    assert metrics["actor/opd_log_ratio_clipfrac"] == 0.0
    objective.backward()
    assert student_logp.grad.item() == pytest.approx(20.0)
