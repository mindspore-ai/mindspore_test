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
import os
import pytest
import stat
import mindspore as ms
import mindspore.nn as nn


def remove_ckpt(file_name):
    """remove ckpt."""
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_save_and_load_sft_with_crc():
    """
    Feature: save and load checkpoint
    Description: test ms.save_checkpoint and ms.load_checkpoint with crc_check
    Expectation: success
    """
    net = nn.Dense(2, 2)
    ckpt_path = "checkpoint_1.safetensors"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net.parameters_dict(), ckpt_path, format="safetensors", crc_check=True)
    ms.load_checkpoint(ckpt_path, format="safetensors", crc_check=True)
    remove_ckpt(ckpt_path)
