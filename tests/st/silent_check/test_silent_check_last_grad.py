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
"""
Test module for testing silent check.
"""
import os
from tests.mark_utils import arg_mark
import pytest


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_silent_check1():
    """
    Feature: Test silent check for last grad node
    Description: Test silent check in non-sink mode
    Expectation: SilentCheckV2 operator is executed
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    py_file = 'silent_check_last_grad1.py'
    ret1 = os.system(f"bash {sh_path}/singlerun_silent_check.sh {sh_path}/{py_file}")
    ret2 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    assert ret1 == 0
    assert ret2 == 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["kbk", "pyboost"])
def test_silent_check2(mode):
    """
    Feature: Test silent check for last grad node
    Description: Test silent check in sink mode
    Expectation: SilentCheckV2 operator is executed
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    py_file = 'silent_check_last_grad2.py'
    ret1 = os.system(f"bash {sh_path}/singlerun_silent_check.sh {sh_path}/{py_file} {mode}")
    ret2 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    assert ret1 == 0
    assert ret2 == 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_silent_check_prod_grad():
    """
    Feature: Test silent check for last grad node of prod op
    Description: Test silent check for last grad node of prod op
    Expectation: No error occurs
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    py_file = 'prod_grad.py'
    ret1 = os.system(f"bash {sh_path}/singlerun_silent_check.sh {sh_path}/{py_file}")
    assert ret1 == 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')
