# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.identity = ops.Identity()

    def construct(self, x):
        return self.identity(x)


def generate_testcases(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(3, 4, 5, 6).astype(nptype)
    net = Net()
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.random.randn(3, 4, 5, 6).astype(nptype)
    net = Net()
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_float64():
    generate_testcases(np.float64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_float32():
    generate_testcases(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_float16():
    generate_testcases(np.float16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_uint64():
    generate_testcases(np.uint64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_int64():
    generate_testcases(np.int64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_uint32():
    generate_testcases(np.uint32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_int32():
    generate_testcases(np.int32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_uint16():
    generate_testcases(np.uint16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_int16():
    generate_testcases(np.int16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_uint8():
    generate_testcases(np.uint8)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_int8():
    generate_testcases(np.int8)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_bool():
    generate_testcases(np.bool_)
