# Copyright 2021-2024 Huawei Technologies Co., Ltd
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

import json
import os
import shutil
import sys
import tempfile
import numpy as np

import mindspore as ms
from mindspore import context
from mindspore import ops, nn, Tensor
from mindspore.ops.operations.math_ops import NPUClearFloatStatusV2

from dump_test_utils import check_dump_structure
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap

e2e_async_dump_json = {
    "common_dump_settings": {
        "dump_mode": 0,
        "path": "",
        "net_name": "Net",
        "iteration": "0",
        "input_output": 0,
        "kernels": ["Default/Conv-op12"],
        "support_device": [0, 1, 2, 3, 4, 5, 6, 7],
        "op_debug_mode": 0
    },
    "e2e_dump_settings": {
        "enable": False,
        "trans_flag": True
    }
}


class OverflowNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.clear_status = NPUClearFloatStatusV2()
        self.sub = ops.Sub()
        self.neg = ops.Neg()

    def construct(self, x):
        init = Tensor([0] * 8, dtype=ms.int32)
        clear_status = self.clear_status(init)
        x = ops.depend(x, clear_status)
        res = self.sub(x, self.neg(x))

        return res


def generate_async_overflow_dump_json(dump_path, json_file_name):
    json_data = e2e_async_dump_json
    json_data["common_dump_settings"]["op_debug_mode"] = 3
    json_data["common_dump_settings"]["path"] = dump_path
    with open(json_file_name, 'w') as f:
        json.dump(json_data, f)


def run_trans_flag(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        if test_name == "test_async_overflow_dump":
            generate_async_overflow_dump_json(dump_path, dump_config_path)

        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        if test_name == "test_async_overflow_dump":
            net = OverflowNet()
            value = 65534
            data = np.full((2, 3), value, dtype=np.float16)
            predict = Tensor(data, dtype=ms.float16)
            net(predict)
        check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
        dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')

        assert os.path.exists(dump_data_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_ascend_async_overflow():
    """
    Feature: Ascend async overflow dump with self defined net.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_async_overflow_dump")
