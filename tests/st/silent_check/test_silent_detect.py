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

import math
import os
import tempfile
import re
import mindspore as ms
from mindspore import Tensor, nn, ops, mutable
from mindspore._c_expression import get_silent_detect_feature_name, get_silent_detect_config
from mindspore import jit

from tests.mark_utils import arg_mark


def extract_silent_detect_log(log_content):
    lines = log_content.split("\n")
    silent_detect_log = []

    def remove_timestamp(text):
        return re.sub(r'timestamp:\s*\d+\s*,?\s*', '', text)

    for line in lines:
        if "[V13001]" in line:
            arr = line.split(']')
            assert arr
            text = arr[-1].strip()
            if text.startswith(("SilentDetect initialized with enable",
                                "Silent detect receives data",
                                "After silent detect",
                                "Skip the unsupported data type",
                                "Strike happened.",
                                "Strike will not be recorded")):
                silent_detect_log.append(remove_timestamp(text))
    return silent_detect_log


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tensordump = ops.TensorDump()
        self.names = ["layernorm", "embedding", "transformer", "linear"]

    @jit
    def construct(self, inputs):
        for idx, x in enumerate(inputs):
            name = self.names[idx]
            self.tensordump(get_silent_detect_feature_name(name), x)
        outputs = ops.clip_by_value(inputs, 0, 1)
        return outputs


def test_silent_detect_config_parser_sub():
    """
    Feature: Test silent detect with MS_NPU_ASD_CONFIG subprocess.
    Description: Test silent detect with MS_NPU_ASD_CONFIG subprocess.
    Expectation: Successfully run the silent detect with MS_NPU_ASD_CONFIG subprocess.
    """
    config_dict = {'cooldown': 1, 'strikes_num': 5, 'strikes_window': 100, 'checksum_cooldown': 10,
                   'upper_thresh1': 60, 'upper_thresh2': 600, 'grad_sample_interval': 30}
    config_str = "enable:true,with_checksum:true"
    for k, v in config_dict.items():
        config_str += f",{k}:{v}"
    os.environ['MS_NPU_ASD_CONFIG'] = config_str
    vals = [0.1, 10, math.inf, math.nan, 0.5]

    for k, v in config_dict.items():
        res = get_silent_detect_config(k)
        assert res == v, f"res not equal to v, res = {res}, v = {v}"
    net = Net()
    for val in vals:
        inputs = mutable([Tensor(val, ms.float32), Tensor(
            val, ms.float16), Tensor(val, ms.bfloat16), Tensor(val, ms.int32)])
        out = net(inputs)
        print("out is ", out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_silent_detect_v4():
    """
    Feature: Test silent detect with MS_NPU_ASD_CONFIG.
    Description: Test silent detect with MS_NPU_ASD_CONFIG.
    Expectation: The output of silent detect matches the expected output in the log file.
    """
    os.environ["VLOG_v"] = "13001"
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    with open(os.path.join(current_dir_path, "silent_detect_target.log"), 'r', encoding='utf-8') as file:
        target_out = [line.strip() for line in file if line.strip()]

    with tempfile.NamedTemporaryFile(mode='w+', delete=True, encoding='utf-8') as temp_file:
        full_command = f"pytest -s test_silent_detect.py::test_silent_detect_config_parser_sub > {temp_file.name} 2>&1"
        exit_code = os.system(full_command)
        temp_file.seek(0)
        log_content = temp_file.read()
        assert exit_code == 0

        silent_detect_log = extract_silent_detect_log(log_content)
        assert target_out == silent_detect_log
    del os.environ["VLOG_v"]
