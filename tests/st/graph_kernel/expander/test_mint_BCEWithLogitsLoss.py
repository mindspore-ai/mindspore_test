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
import numpy as np
import mindspore.context as context
import mindspore
from mindspore import Tensor, mint
from mindspore.nn import Cell


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.binary_cross_entropy_with_logits = mint.nn.functional.binary_cross_entropy_with_logits

    def construct(self, x1, x2, weight, pos_weight, mode):
        return self.binary_cross_entropy_with_logits(x1, x2, weight, mode, pos_weight)


def get_output(x1, x2, weight, mode, pos_weight, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(jit_level='O1', graph_kernel_flags="--dump_as_text --enable_expand_ops=BCEWithLogitsLoss")
    else:
        context.set_context(jit_level='O0')
    net = Net()
    output = net(x1, x2, weight, pos_weight, mode)
    output = output.asnumpy()
    if enable_graph_kernel:
        context.set_context(graph_kernel_flags="")
    return output


def run_basic(dtype, mode='mean', compare_precision=1e-4):
    def _remove_file(file_path):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

    np.random.seed(1)
    x1 = Tensor(np.random.normal(
        0.5, 0.01, [256, 256]).astype(np.float32), dtype=dtype)
    x2 = Tensor(np.random.normal(
        0.5, 0.01, [256, 256]).astype(np.float32), dtype=dtype)
    weight = Tensor(np.random.normal(
        0.5, 0.01, [256, 256]).astype(np.float32), dtype=dtype)
    pos_weight = Tensor(np.random.normal(
        0.5, 0.01, [256, 256]).astype(np.float32), dtype=dtype)
    expect = get_output(x1, x2, weight, mode, pos_weight, False)
    output = get_output(x1, x2, weight, mode, pos_weight, True)
    dump_file = "./graph_kernel_dump/dvm_kernel_{}.txt".format(os.getpid())
    try:
        np.testing.assert_allclose(expect, output, compare_precision, compare_precision)
    except Exception as ex:
        if os.path.isfile(dump_file):
            print("dump_file", dump_file)
            with open(dump_file, 'r') as f:
                for line in f:
                    print(line)
            _remove_file(dump_file)
        raise RuntimeError("Precision compare failed!\n{}".format(ex))
    _remove_file(dump_file)


def test_basic_ascend_f16():
    """
    Feature: test graph kernel mint.BCEWithLogitsLoss
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(mindspore.float16, compare_precision=1e-3)


def test_basic_ascend_f32():
    """
    Feature: test graph kernel mint.BCEWithLogitsLoss
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(mindspore.float32)


def test_basic_ascend_f32_none():
    """
    Feature: test graph kernel mint.BCEWithLogitsLoss
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(mindspore.float32, 'none')


def test_basic_ascend_f32_sum():
    """
    Feature: test graph kernel mint.BCEWithLogitsLoss
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(mindspore.float32, 'sum')
