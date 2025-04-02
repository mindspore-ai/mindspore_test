# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import pytest

from tests.st.mindrlhf.utils import check_log

root_path = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_qwen_grpo():
    """
    Feature: test Qwen GRPO training
    Description: test Qwen GRPO training
    Expectation: success
    """
    os.system(f"bash {root_path}/run_qwen_grpo_test.sh")

    log_path = f"{root_path}/qwen2_one_log/worker_0.log"
    check_pair = {"Save checkpoints in": 1}

    check_log(log_path, check_pair)
