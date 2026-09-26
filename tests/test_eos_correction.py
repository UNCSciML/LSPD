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

import torch

from verl.utils.torch_functional import get_response_mask, remap_teacher_eos_logits


def test_remap_teacher_eos_logits_moves_probability_mass_to_student_eos():
    teacher_eos_token_id = 1
    student_eos_token_id = 3
    logits = torch.tensor([[0.2, 1.4, -0.7, 0.5, 0.0]], dtype=torch.float32)
    original_probs = logits.softmax(dim=-1)

    remapped_probs = remap_teacher_eos_logits(
        logits.clone(),
        teacher_eos_token_id=teacher_eos_token_id,
        student_eos_token_id=student_eos_token_id,
    ).softmax(dim=-1)

    torch.testing.assert_close(
        remapped_probs[..., student_eos_token_id],
        original_probs[..., student_eos_token_id] + original_probs[..., teacher_eos_token_id],
    )
    assert remapped_probs[..., teacher_eos_token_id].item() < 1e-40
    for token_id in {0, 2, 4}:
        torch.testing.assert_close(remapped_probs[..., token_id], original_probs[..., token_id])


def test_remap_teacher_eos_logits_is_noop_for_shared_eos():
    logits = torch.tensor([[0.2, 1.4, -0.7]], dtype=torch.float32)
    output = remap_teacher_eos_logits(
        logits,
        teacher_eos_token_id=1,
        student_eos_token_id=1,
    )
    torch.testing.assert_close(output, logits)


def test_response_mask_includes_first_eos_of_either_model():
    responses = torch.tensor([[12, 151645, 151643, 0], [12, 13, 151643, 0]])
    mask = get_response_mask(responses, eos_token=[151645, 151643])
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 0]]
