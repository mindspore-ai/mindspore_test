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
# pylint: disable=redefined-builtin
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, mint, Tensor
from mindspore.ops.composite import GradOperation
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


class SoftsignNet(nn.Cell):

    def __init__(self):
        super(SoftsignNet, self).__init__()
        self.net = mint.nn.Softsign()

    def construct(self, input):
        return self.net(input)


class SoftsignGradNet(nn.Cell):

    def __init__(self, net):
        super(SoftsignGradNet, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.net = net

    def construct(self, input):
        return self.grad(self.net)(input)


def softsign_forward_func(*inputs):
    out = SoftsignNet()(*inputs)
    return out


def softsign_backward_func(*inputs):
    grad = SoftsignGradNet(SoftsignNet())(*inputs)
    return grad


@ops_binary_cases(
    OpsBinaryCase(input_info=[((10, 10), np.float32)],
                  output_info=[((10, 10), np.float32), ((10, 10), np.float32)],
                  extra_info='softsign'))
def mint_nn_softsign_binary_case1(input_binary_data=None,
                                  output_binary_data=None):
    output = softsign_forward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = softsign_backward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output[0].asnumpy(), output_binary_data[1], 1e-04,
                       1e-04)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_softsign(mode):
    """
    Feature: softsign
    Description: Verify the result of softsign.
    Expectation: success
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(jit_config={"jit_level": "O0"},
                               mode=ms.GRAPH_MODE)
    mint_nn_softsign_binary_case1()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_softsign_dyn():
    """
    Feature: Dynamic shape of softsign
    Description: test softsign with dynamic rank/shape.
    Expectation: success
    """
    input1 = generate_random_input((10, 10), np.float32)
    input2 = generate_random_input((5, 5, 5), np.float32)
    TEST_OP(mint.nn.functional.softsign,
            [[ms.Tensor(input1)], [ms.Tensor(input2)]],
            "softsign",
            disable_mode=["GRAPH_MODE"])
