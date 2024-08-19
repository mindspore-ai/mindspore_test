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

import os
import sys
import tempfile
import glob
import shutil
import numpy as np
import mindspore
import mindspore.context as context
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from dump_test_utils import generate_dump_json, check_dump_structure

class ConvNet(nn.Cell):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv2 = ops.Conv2D(out_channel=3, kernel_size=1)

    def construct(self, x, weight):
        return self.conv2(x, weight)


class NetMulAdd(nn.Cell):
    def __init__(self):
        super(NetMulAdd, self).__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x_, y_):
        x_ = self.mul(x_, 2)
        y_ = self.mul(y_, 2)
        x_ = self.add(x_, y_)
        y_ = self.add(x_, y_)
        return self.add(x_, y_)


def run_trans_flag(test_name):
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, test_name)
        dump_config_path = os.path.join(tmp_dir, '{}.json'.format(test_name))
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)

        if test_name == "test_e2e_dump_trans_true_op_debug_mode":
            generate_dump_json(dump_path, dump_config_path, test_name)
            tensor = Tensor(np.full((1, 3, 3, 3), 65504, dtype=np.float16), mindspore.float16)
            weight = Tensor(np.full((3, 3, 1, 1), 65504, dtype=np.float16), mindspore.float16)
            net = ConvNet()
            expect = net(tensor, weight)
            check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
            dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
            assert os.path.exists(dump_data_path)
            # tensor data in host format.
            output_name = "Conv2D.Default_Conv2D-op*.output.0.DefaultFormat.*.npy"
            output_path = glob.glob(os.path.join(dump_data_path, output_name))[0]
            real_path = os.path.realpath(output_path)
            output = np.load(real_path)
            assert output.shape == (1, 3, 3, 3)
            assert np.array_equal(output, expect)

        if test_name == "test_e2e_dump_set_overflow_number":
            set_overflow_num = 2
            generate_dump_json(dump_path, dump_config_path, test_name, overflow_number=set_overflow_num)
            data = np.array([60000, 60000]).astype(np.float16)
            net = NetMulAdd()
            net(Tensor(data), Tensor(data))
            check_dump_structure(dump_path, dump_config_path, 1, 0, 1)
            dump_data_path = os.path.join(dump_path, 'rank_0', 'Net', '0', '0')
            assert os.path.exists(dump_data_path)
            overflow_files = glob.glob(os.path.join(dump_data_path, "*.npy"))
            overflow_files_num = len(overflow_files)
            assert overflow_files_num == set_overflow_num * 3

        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_ascend_kernel_by_kernel_trans_true_op_debug_mode():
    """
    Feature: Ascend kernel by kernel dump with overflow.
    Description: Test kernel by kernel dump in Ascend with trans_flag is configured to true.
    Expectation: Dump files has tensor data in host format (4 dimensions).
    """
    context.set_context(jit_level='O0')
    os.environ['INF_NAN_MODE_ENABLE'] = "1"
    os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE'] = "INFNAN_MODE"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_trans_true_op_debug_mode")
    del os.environ['INF_NAN_MODE_ENABLE']
    del os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_e2e_dump_set_overflow_number():
    """
    Feature: The number of overflow dump during the training process can be configured.
    Description: Test kernel by kernel dump in Ascend with overflow_number is configured.
    Expectation: The number of dump files matches the value of the overflow_number parameter that has been set.
    """
    context.set_context(jit_level='O0')
    os.environ['INF_NAN_MODE_ENABLE'] = "1"
    os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE'] = "INFNAN_MODE"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_trans_flag("test_e2e_dump_set_overflow_number")
    del os.environ['INF_NAN_MODE_ENABLE']
    del os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE']
