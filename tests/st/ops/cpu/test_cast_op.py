# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


class Net(Cell):
    def __init__(self, dtype):
        super(Net, self).__init__()
        self.Cast = P.Cast()
        self.dtype = dtype

    def construct(self, x):
        return self.Cast(x, self.dtype)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_complex64():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type complex64
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.complex64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'complex64'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_complex128():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type complex128
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.complex128

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'complex128'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_bool():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type bool
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.bool_

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'bool'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_float16():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type float16
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.float16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'float16'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_float32():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type float32
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.float32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'float32'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_float64():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type float64
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.float64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'float64'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_bfloat16():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type bfloat16
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.bfloat16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.dtype == mstype.bfloat16


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_int8():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type int8
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.int8

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'int8'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_int16():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type int16
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.int16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'int16'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_int32():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type int32
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.int32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'int32'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_int64():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type int64
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.int64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'int64'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_uint8():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type uint8
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.uint8

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'uint8'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_uint16():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type uint16
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.uint16

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'uint16'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_uint32():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type uint32
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.uint32

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'uint32'


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cast_uint64():
    """
    Feature: cast
    Description: Verify the result of cast with dst_type uint64
    Expectation: success
    """
    tensor_to_cast = []
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.bool_)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.float64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.int64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint8)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint16)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint32)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.uint64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex64)))
    tensor_to_cast.append(Tensor(np.random.uniform(-2, 2, (3, 2)).astype(np.complex128)))
    t = mstype.uint64

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    for tensor in tensor_to_cast:
        net = Net(t)
        output = net(tensor)
        assert output.asnumpy().dtype == 'uint64'
