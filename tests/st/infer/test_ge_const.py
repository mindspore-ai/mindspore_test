# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test GE const"""

import pytest
import os

def my_assert(condition, log_path):
    if condition:
        assert True
    else:
        os.system(f"cat {log_path}")
        assert False

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_const_log():
    """
    Feature: GE Const
    Description: check whether the parameter is convert to GE Const op or not by checking the info log
    Expectation: success
    """
    case_name = "test_const_log"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    log_path = f"{sh_path}/{case_name}.log"
    ret = os.system(
        f"bash {sh_path}/run_with_log.sh 1 {sh_path}/{case_name}.py &> {log_path}")
    my_assert(ret == 0, log_path)

    find_keyword1 = False
    key_word1 = "InitParam with const for node"
    key_word2 = "Start AllocConstMemory, memory_size:"
    with open(log_path) as f:
        line = f.readline()
        while line:
            if not find_keyword1:
                if key_word1 in line:
                    find_keyword1 = True
            else:
                if key_word2 in line:
                    memory_size = int(line.split(":")[-1])
                    # the shape of parameter is (16, 512)
                    my_assert(memory_size > (16 * 512), log_path)
            line = f.readline()
    my_assert(find_keyword1, log_path)
    os.system(f"rm -rf {log_path}")
