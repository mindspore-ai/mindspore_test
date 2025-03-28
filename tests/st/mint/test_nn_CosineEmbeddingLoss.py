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
import mindspore.mint.nn as mnn
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def cosine_embedding_loss_forward_func(input1,
                                       input2,
                                       target,
                                       margin=0.0,
                                       reduction="mean"):
    return mnn.functional.cosine_embedding_loss(input1, input2, target, margin,
                                                reduction)


@test_utils.run_with_cell
def cosine_embedding_loss_backward_func(input1,
                                        input2,
                                        target,
                                        margin=0.0,
                                        reduction="mean"):
    return ms.grad(cosine_embedding_loss_forward_func,
                   (0, 1))(input1, input2, target, margin, reduction)


@ops_binary_cases(
    OpsBinaryCase(input_info=[((10, 10), np.float32), ((10, 10), np.float32),
                              ((10,), np.int64)],
                  output_info=[((10,), np.float32), ((10, 10), np.float32),
                               ((10, 10), np.float32)],
                  extra_info='cosine_embedding_loss'))
def mint_nn_cosine_embedding_loss_binary_case1(input_binary_data=None,
                                               output_binary_data=None):
    output = cosine_embedding_loss_forward_func(Tensor(input_binary_data[0]),
                                                Tensor(input_binary_data[1]),
                                                Tensor(input_binary_data[2]),
                                                margin=0.0,
                                                reduction='none')
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = cosine_embedding_loss_backward_func(Tensor(input_binary_data[0]),
                                                 Tensor(input_binary_data[1]),
                                                 Tensor(input_binary_data[2]),
                                                 margin=0.0,
                                                 reduction='none')
    assert np.allclose(output[0].asnumpy(), output_binary_data[1], 1e-04,
                       1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[2], 1e-04,
                       1e-04)


@ops_binary_cases(
    OpsBinaryCase(input_info=[((10, 10), np.float32), ((10, 10), np.float32),
                              ((10,), np.int64)],
                  output_info=[((), np.float32), ((10, 10), np.float32),
                               ((10, 10), np.float32)],
                  extra_info='cosine_embedding_loss'))
def mint_nn_cosine_embedding_loss_binary_case2(input_binary_data=None,
                                               output_binary_data=None):
    output = cosine_embedding_loss_forward_func(Tensor(input_binary_data[0]),
                                                Tensor(input_binary_data[1]),
                                                Tensor(input_binary_data[2]),
                                                margin=0.5,
                                                reduction='mean')
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = cosine_embedding_loss_backward_func(Tensor(input_binary_data[0]),
                                                 Tensor(input_binary_data[1]),
                                                 Tensor(input_binary_data[2]),
                                                 margin=0.5,
                                                 reduction='mean')
    assert np.allclose(output[0].asnumpy(), output_binary_data[1], 1e-04,
                       1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[2], 1e-04,
                       1e-04)


@ops_binary_cases(
    OpsBinaryCase(input_info=[((10, 10), np.float32), ((10, 10), np.float32),
                              ((10,), np.int64)],
                  output_info=[((), np.float32), ((10, 10), np.float32),
                               ((10, 10), np.float32)],
                  extra_info='cosine_embedding_loss'))
def mint_nn_cosine_embedding_loss_binary_case3(input_binary_data=None,
                                               output_binary_data=None):
    output = cosine_embedding_loss_forward_func(Tensor(input_binary_data[0]),
                                                Tensor(input_binary_data[1]),
                                                Tensor(input_binary_data[2]),
                                                margin=-0.5,
                                                reduction='sum')
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = cosine_embedding_loss_backward_func(Tensor(input_binary_data[0]),
                                                 Tensor(input_binary_data[1]),
                                                 Tensor(input_binary_data[2]),
                                                 margin=-0.5,
                                                 reduction='sum')
    assert np.allclose(output[0].asnumpy(), output_binary_data[1], 1e-04,
                       1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[2], 1e-04,
                       1e-04)


@ops_binary_cases(
    OpsBinaryCase(input_info=[((10,), np.float32), ((10,), np.float32),
                              ((), np.int64)],
                  output_info=[((), np.float32), ((10,), np.float32),
                               ((10,), np.float32)],
                  extra_info='cosine_embedding_loss'))
def mint_nn_cosine_embedding_loss_binary_case4(input_binary_data=None,
                                               output_binary_data=None):
    output = cosine_embedding_loss_forward_func(Tensor(input_binary_data[0]),
                                                Tensor(input_binary_data[1]),
                                                Tensor(input_binary_data[2]),
                                                margin=-0.5,
                                                reduction='sum')
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = cosine_embedding_loss_backward_func(Tensor(input_binary_data[0]),
                                                 Tensor(input_binary_data[1]),
                                                 Tensor(input_binary_data[2]),
                                                 margin=-0.5,
                                                 reduction='sum')
    assert np.allclose(output[0].asnumpy(), output_binary_data[1], 1e-04,
                       1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[2], 1e-04,
                       1e-04)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cosine_embedding_loss_func_normal(mode):
    """
    Feature: cosine_embedding_loss
    Description: Verify the result of cosine_embedding_loss.
    Expectation: success
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(jit_config={"jit_level": "O0"},
                               mode=ms.GRAPH_MODE)
    mint_nn_cosine_embedding_loss_binary_case1()
    mint_nn_cosine_embedding_loss_binary_case2()
    mint_nn_cosine_embedding_loss_binary_case3()
    mint_nn_cosine_embedding_loss_binary_case4()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_cosine_embedding_loss_dyn():
    """
    Feature: Dynamic shape of cosine_embedding_loss
    Description: test cosine_embedding_loss with dynamic rank/shape.
    Expectation: success
    """
    x1 = generate_random_input((10, 10), np.float32)
    x2 = generate_random_input((10, 10), np.float32)
    x3 = 2 * np.random.randint(0, 2, size=10) - 1

    y1 = generate_random_input((10,), np.float32)
    y2 = generate_random_input((10,), np.float32)
    y3 = np.array(-1)
    TEST_OP(mnn.functional.cosine_embedding_loss, [[
        ms.Tensor(x1),
        ms.Tensor(x2),
        ms.Tensor(x3), 0.0, 'mean'
    ], [ms.Tensor(y1), ms.Tensor(y2),
        ms.Tensor(y3), 0.5, 'sum']],
            "cosine_embedding_loss",
            disable_mode=["GRAPH_MODE"])
