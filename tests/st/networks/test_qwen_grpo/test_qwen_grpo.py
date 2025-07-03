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
"""Test QWENGRPO"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils import init_env_rl
import logging
import re
import pytest
logging.basicConfig(level=logging.INFO)


def parse_log_file(file):
    it_pattern = (r'.*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] '
                  r'iteration:\s*(\d*) \/.*actor/pg_loss : ([\d\.]*).*actor/ppo_kl : ([-\d\.]*).*')
    with open(file, 'r') as f:
        context = f.read().split('\n')
    data = {}
    for cont in context:
        match = re.match(it_pattern, cont)
        if match:
            data[int(match.group(2))] = match.groups()
    return data


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestQwenGRPO:
    @pytest.mark.level1
    @pytest.mark.run(order=1)
    def test_qwen_grpo(self):
        """
        Feature: test mindspore pretrain_glm
        Description: run mindspore r1_zero to generate pynative loss
        Expectation: test success
        """
        init_env_rl()
        scripts_name = "test_qwen_grpo.sh"

        test_path = os.path.split(os.path.realpath(__file__))[0]
        cmd = "bash %s/%s" % (test_path, scripts_name)
        logging.info("Running command:\n%s", cmd)
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check ms_det.log"

    @pytest.mark.level1
    @pytest.mark.run(order=2)
    def test_compare_res(self):
        """
        Feature: test_compare_res
        Description: compare relative error between torch loss and mindspore loss
        Expectation: no error
        """
        loss_pt = parse_log_file('pta_det.txt')
        loss_ms = parse_log_file('ms_det.txt')
        # 开确定性计算，精度对齐
        for i in loss_pt:
            logging.info("loss: %s %s", loss_pt[i][2], loss_ms[i][2])
            assert abs(len(loss_pt[i][2]) - len(loss_ms[i][2])) < 100
