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

from concurrent.futures import ThreadPoolExecutor

from verl.utils.reward_score.ttrl_math import reward_func


def test_ttrl_reward_supports_high_recall_grading_from_worker_thread():
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            reward_func,
            "math_dapo",
            r"Therefore the answer is \boxed{\frac{1}{2}}.",
            "0.5",
        ).result(timeout=15)

    assert result["score"] == 1.0
    assert result["format_score"] == 1.0
    assert result["acc"] is True


def test_ttrl_reward_still_requires_boxed_answer_from_worker_thread():
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(reward_func, "AIME24", "The answer is 204.", "204").result(timeout=15)

    assert result["score"] == 0.0
    assert result["format_score"] == 0.0
    assert result["acc"] is False
