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
# pylint: disable=unused-variable

import pytest
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import mint
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


class Net(ms.nn.Cell):
    def construct(self, *size, dtype=None, device=None):
        return mint.empty(*size, dtype=dtype, device=device)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_normal1(mode):
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_size = (1, 2, 3)
    dtype = mstype.float32

    net = Net()
    y = net(input_size)
    assert np.allclose(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_empty_normal2():
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_size = (1, 2, 3)
    dtype = mstype.float32

    net = Net()
    y = net(input_size, device="CPU")
    assert np.allclose(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_normal3(mode):
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_size = (1, 2, 3)
    dtype = mstype.float64

    net = Net()
    y = net(input_size, dtype=dtype, device="Ascend")
    assert np.allclose(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_normal4(mode):
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_size = (1, 2, 3, 4, 5, 6, 7)
    dtype = mstype.int64

    net = Net()
    y = net(input_size, dtype=dtype)
    assert np.allclose(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_empty_normal5(mode):
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = mstype.float32

    net = Net()
    y = net(1, 2, 3)
    assert np.allclose(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, dtype)

def empty_forward_func_dyn_test(input_size, dtype=None):
    y = Net()(input_size, dtype=dtype)
    return y.shape

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_empty_dynamic_shape():
    """
    Feature: Test empty with dynamic shape in graph mode.
    Description: call ops.extend.empty with valid input and index.
    Expectation: return the correct value.
    """
    TEST_OP(empty_forward_func_dyn_test, [[(2, 3)], [(3, 4, 5)]], '', disable_yaml_check=True,
            disable_grad=True, disable_mode=['GRAPH_MODE'])
