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
# pylint: disable=unused-variable
# pylint: disable=redefined-builtin
# pylint: disable=W0235
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, mint
from mindspore.ops.composite import GradOperation
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase


class MatMulNet(nn.Cell):
    def __init__(self):
        super(MatMulNet, self).__init__()

    def construct(self, input, other):
        out = mint.matmul(input, other)
        return out


class MatMulGradNet(nn.Cell):
    def __init__(self, net):
        super(MatMulGradNet, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.net = net

    def construct(self, input, other):
        return self.grad(self.net)(input, other)


def matmul_forward_func(*inputs):
    out = MatMulNet()(*inputs)
    return out


def matmul_backward_func(*inputs):
    grads = MatMulGradNet(MatMulNet())(*inputs)
    return grads


def mint_matmul_binary_compare(input_binary_data, output_binary_data, loss, is_bfloat16=False):

    if is_bfloat16:
        inputs = [ms.Tensor(np_data, ms.bfloat16) for np_data in input_binary_data]
    else:
        inputs = [ms.Tensor(np_data) for np_data in input_binary_data]

    out_forward = matmul_forward_func(*inputs)
    expect_out = output_binary_data[0]
    if is_bfloat16:
        np.testing.assert_allclose(out_forward.float().asnumpy(), expect_out, rtol=loss, atol=loss)
    else:
        np.testing.assert_allclose(out_forward.asnumpy(), expect_out, rtol=loss, atol=loss)

    grads = matmul_backward_func(*inputs)
    expect_grads = output_binary_data[1:]
    for idx, expect_grad in enumerate(expect_grads):
        if is_bfloat16:
            np.testing.assert_allclose(grads[idx].float().asnumpy(), expect_grad, rtol=loss, atol=loss)
        else:
            np.testing.assert_allclose(grads[idx].asnumpy(), expect_grad, rtol=loss, atol=loss)


@ops_binary_cases(OpsBinaryCase(input_info=[((3, 3), np.float32), ((3, 1024), np.float32)],
                                output_info=[((3, 1024,), np.float32), ((3, 3), np.float32), ((3, 1024), np.float32)],
                                extra_info='auto_drive'))
def mint_matmul_binary_case1(input_binary_data=None, output_binary_data=None):
    mint_matmul_binary_compare(input_binary_data, output_binary_data, loss=1e-4)


@ops_binary_cases(OpsBinaryCase(input_info=[((4096, 5120), np.float32), ((5120, 640), np.float32)],
                                output_info=[((4096, 640), np.float32), ((4096, 5120), np.float32),
                                             ((5120, 640), np.float32)],
                                extra_info='auto_drive'))
def mint_matmul_binary_case2(input_binary_data=None, output_binary_data=None):
    mint_matmul_binary_compare(input_binary_data, output_binary_data, loss=4e-3, is_bfloat16=True)


@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_matmul_binary_cases(mode):
    """
    Feature: standard forward, backward features.
    Description: test mint matmul
    Expectation: expect correct result.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    mint_matmul_binary_case1()
    mint_matmul_binary_case2()
