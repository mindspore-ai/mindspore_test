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
# pylint: disable=redefined-builtin
# pylint: disable=W0235
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops.composite import GradOperation
from tests.mark_utils import arg_mark
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase


class TriangularSolveTensorNet(nn.Cell):

    def __init__(self):
        super(TriangularSolveTensorNet, self).__init__()

    def construct(self,
                  b,
                  A,
                  upper=True,
                  transpose=False,
                  unitriangular=False):
        out = b.triangular_solve(A, upper, transpose, unitriangular)
        return out


class TriangularSolveGradNet(nn.Cell):

    def __init__(self, net):
        super(TriangularSolveGradNet, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.net = net

    def construct(self,
                  b,
                  A,
                  upper=True,
                  transpose=False,
                  unitriangular=False):
        return self.grad(self.net)(b, A, upper, transpose, unitriangular)


def triangular_solve_forward_tensor(*inputs):
    out = TriangularSolveTensorNet()(*inputs)
    return out


def triangular_solve_backward_tensor(*inputs):
    grads = TriangularSolveGradNet(TriangularSolveTensorNet())(*inputs)
    return grads


@ops_binary_cases(
    OpsBinaryCase(input_info=[((4, 3), np.float32), ((4, 4), np.float32)],
                  output_info=[((4, 3), np.float32), ((4, 4), np.float32),
                               ((4, 3), np.float32), ((4, 4), np.float32)],
                  extra_info='triangular_solve'))
def tensor_triangular_solve_binary_case1(input_binary_data=None,
                                         output_binary_data=None):
    output = triangular_solve_forward_tensor(Tensor(input_binary_data[0]),
                                             Tensor(input_binary_data[1]))
    assert np.allclose(output[0].asnumpy(), output_binary_data[0], 1e-04,
                       1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[1], 1e-04,
                       1e-04)
    output = triangular_solve_backward_tensor(Tensor(input_binary_data[0]),
                                              Tensor(input_binary_data[1]))
    assert np.allclose(output[0].asnumpy(), output_binary_data[2], 1e-04,
                       1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[3], 1e-04,
                       1e-04)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_triangular_solve(mode):
    """
    Feature: triangular_solve
    Description: Verify the result of triangular_solve.
    Expectation: success
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(jit_config={"jit_level": "O0"},
                               mode=ms.GRAPH_MODE)
    tensor_triangular_solve_binary_case1()
