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
"""
Test module for testing resume training from specified checkpoint.
How to run this:
    pytest tests/st/networks/large_models/resume/test_parallel_resume.py
"""

import os

from tests.mark_utils import arg_mark
from resume_train_utils import extract_loss_values


class TestResumeTraining:
    """A test class for testing pipeline."""

    @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test parallel trainer for train.
        Expectation: AssertionError
        """
        ascend_home_path = os.getenv('ASCEND_HOME_PATH')
        if not ascend_home_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(f"export MS_DEV_P2P_HCCL_BUFFSIZE=24 && "
                        f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && "
                        f"bash {sh_path}/msrun_launch.sh 4")

        os.system("grep -E 'ERROR|error' {sh_path}/msrun_log/worker_3.log -C 3")
        assert ret == 0

        loss = extract_loss_values("msrun_log/worker_3.log")

        resume_start = 257
        train_middle = 128
        for i in range(0, 10):
            assert abs(loss[resume_start + i] - loss[train_middle + i]) < 0.005
