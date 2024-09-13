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
import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor


def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float16)


def addmm__dyn_shape_func(input1, mat1, mat2, beta, alpha):
    return input1.addmm_(mat1, mat2, beta=beta, alpha=alpha)


def generate_numpy_output(input1, mat1, mat2, beta, alpha):
    return input1 * beta + alpha * np.dot(mat1, mat2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('ms_type', [mstype.float32, mstype.float16])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_addmm(context_mode, ms_type):
    """
    Feature: pyboost function.
    Description: test function inplace_addmm backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    input_np = np.arange(9).astype(np.float32).reshape((3, 3))
    mat1_np = np.arange(12).astype(np.float32).reshape((3, 4))
    mat2_np = np.arange(12).astype(np.float32).reshape((4, 3))
    input1 = Tensor(input_np, dtype=ms_type)
    mat1 = Tensor(mat1_np, dtype=ms_type)
    mat2 = Tensor(mat2_np, dtype=ms_type)

    input1.addmm_(mat1, mat2, beta=0.5, alpha=2)
    expect_output = generate_numpy_output(input_np, mat1_np, mat2_np, 0.5, 2)
    assert np.allclose(input1.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_inplace_addmm_dynamic():
    """
    Feature: pyboost function.
    Description: test function inplace_addmm forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_dyn = ms.Tensor(shape=None, dtype=mstype.float32)
    mat1_dyn = ms.Tensor(shape=None, dtype=mstype.float32)
    mat2_dyn = ms.Tensor(shape=None, dtype=mstype.float32)
    test_cell = test_utils.to_cell_obj(addmm__dyn_shape_func)
    test_cell.set_inputs(input_dyn, mat1_dyn, mat2_dyn, 0.5, 2)

    input_np = np.arange(9).astype(np.float32).reshape((3, 3))
    mat1_np = np.arange(12).astype(np.float32).reshape((3, 4))
    mat2_np = np.arange(12).astype(np.float32).reshape((4, 3))
    input1 = Tensor(input_np, dtype=mstype.float32)
    mat1 = Tensor(mat1_np, dtype=mstype.float32)
    mat2 = Tensor(mat2_np, dtype=mstype.float32)
    test_cell(input1, mat1, mat2, 0.5, 2)
    expect_output = generate_numpy_output(input_np, mat1_np, mat2_np, 0.5, 2)
    assert np.allclose(input1.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
