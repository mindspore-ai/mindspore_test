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

import sys
from typing import Callable, Optional

import numpy as np
import pytest

import mindspore as ms
from tests.st.ops.dynamic_shape import test_op_utils
from tests.st.ops import ops_binary_cases
from tests.st.utils import test_utils

if sys.version_info >= (3, 9):
    list_annotation, tuple_annotation = list, tuple
else:
    from typing import List, Tuple
    list_annotation, tuple_annotation = List, Tuple


@test_utils.run_with_cell
def small_ops_func(x: ms.Tensor, x2: ms.Tensor, trans_x2: bool) -> tuple_annotation[ms.Tensor]:
    gather_out = ms.ops.AllGather()(x)
    output = ms.ops.MatMul(False, trans_x2)(gather_out, x2)
    return output, gather_out


@test_utils.run_with_cell
def fusion_ops_func(
        x: ms.Tensor,
        x2: ms.Tensor,
        group: str,
        world_size: int,
        bias: Optional[ms.Tensor],
        gather_index: int,
        gather_output: bool,
        comm_turn: int,
        trans_input: bool,
        trans_x2: bool,
    ) -> tuple_annotation[ms.Tensor]:
    return ms.ops.all_gather_matmul(
        x, x2, group, world_size, bias, gather_index, gather_output, comm_turn, trans_input, trans_x2
    )


@pytest.mark.parametrize('mode, jit_level, gather_output', [
    (ms.GRAPH_MODE, 'O0', False),
    (ms.GRAPH_MODE, 'O0', True),
    (ms.GRAPH_MODE, 'O1', False),
    (ms.GRAPH_MODE, 'O1', True),
    (ms.GRAPH_MODE, 'O2', True),
    (ms.PYNATIVE_MODE, '', False),
    (ms.PYNATIVE_MODE, '', True),
])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('x2_shape, trans_x2', [((256, 512), False), ((512, 256), True)])
def test_all_gather_matmul_normal(
        mode: int,
        jit_level: str,
        gather_output: bool,
        dtype: ms.dtype,
        x2_shape: tuple_annotation[int],
        trans_x2: bool,
    ) -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
                 mindspore.ops.AllGahter and mindspore.ops.MatMul forword calculation.
    """
    ms.communication.init()
    ms.set_context(mode=mode, device_target='Ascend')
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': jit_level})

    seed = ms.communication.get_rank()
    np.random.seed(seed)
    x = ms.Tensor(test_utils.generate_random_input((128, 256), np.float32)).type(dtype)
    x2 = ms.Tensor(test_utils.generate_random_input(x2_shape, np.float32)).type(dtype)
    # Q: Why use `numpy.random.randn` to generate a random `numpy.ndarray` and then convert it into a
    #    `mindspore.Tensor` instead of directly using `mindspore.ops.StandardNormal` to generate a random
    #    `mindspore.Tensor`?
    # A: Because `mindspore.ops.StandardNormal` does not support the random seed reproduction function on the Ascend
    #    backend, which is not conducive to reproducting results. Reference
    #    https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.StandardNormal.html .

    expected_output, expected_gather_out = small_ops_func(x, x2, trans_x2)
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    world_size = ms.communication.get_group_size()
    actual_output, actual_gather_out = fusion_ops_func(
        x, x2, group, world_size, None, 0, gather_output, 0, False, trans_x2
    )
    assert actual_output.isclose(expected_output, rtol=1e-03, atol=1e-03).all()
    if gather_output:
        assert actual_gather_out.isclose(expected_gather_out, rtol=1e-03, atol=1e-03).all()


def get_dynamic_func(group: str, world_size: int) \
    -> Callable[[ms.Tensor, ms.Tensor, Optional[ms.Tensor], int, bool, int, bool, bool], tuple_annotation[ms.Tensor]]:
    def func(
            x: ms.Tensor,
            x2: ms.Tensor,
            bias: Optional[ms.Tensor],
            gather_index: int,
            gather_output: bool,
            comm_turn: int,
            trans_input: bool,
            trans_x2: bool,
        ) -> tuple_annotation[ms.Tensor]:
        return ms.ops.all_gather_matmul(
            x, x2, group, world_size, bias, gather_index, gather_output, comm_turn, trans_input, trans_x2
        )
    return func


def test_all_gather_matmul_dynamic():
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the dynamic shape function of forward calculation.
    Expectation: The result of forward calculation with inputs in dynamic shapes is equal to the result of forword
                 calculation with inputs in static shapes.
    """
    ms.communication.init()
    ms.set_context(device_target='Ascend')

    seed = ms.communication.get_rank()
    np.random.seed(seed)
    first_input = ms.Tensor(test_utils.generate_random_input((128, 256), np.float16))
    first_x2 = ms.Tensor(test_utils.generate_random_input((256, 512), np.float16))
    second_input = ms.Tensor(test_utils.generate_random_input((256, 512), np.float16))
    second_x2 = ms.Tensor(test_utils.generate_random_input((1024, 512), np.float16))
    # Q: Why use `numpy.random.randn` to generate a random `numpy.ndarray` and then convert it into a
    #    `mindspore.Tensor` instead of directly using `mindspore.ops.StandardNormal` to generate a random
    #    `mindspore.Tensor`?
    # A: Because `mindspore.ops.StandardNormal` does not support the random seed reproduction function on the Ascend
    #    backend, which is not conducive to reproducting results. Reference
    #    https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.StandardNormal.html .
    inputs_seq = [
        [first_input, first_x2, None, 0, True, 0, False, False],
        [second_input, second_x2, None, 0, False, 0, False, True],
    ]

    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    world_size = ms.communication.get_group_size()
    dynamic_func = get_dynamic_func(group, world_size)
    test_op_utils.TEST_OP(
        dynamic_func,
        inputs_seq,
        '',
        disable_input_check=True,
        disable_yaml_check=True,
        disable_mode=['GRAPH_MODE'],
        disable_grad=True,
    )


def binary_compare(
        input_binary_data: list_annotation[np.ndarray],
        output_binary_data: list_annotation[np.ndarray],
        trans_x2: bool,
        gather_output: bool,
    ) -> None:
    x, x2 = test_utils.get_inputs_tensor(input_binary_data)
    if x.dtype == ms.float32:
        x = x.astype(ms.bfloat16)
        x2 = x2.astype(ms.bfloat16)
    expected_output, expected_gather_out = output_binary_data
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    world_size = ms.communication.get_group_size()
    actual_output_tensor, actual_gather_out_tensor = fusion_ops_func(
        x, x2, group, world_size, None, 0, gather_output, 0, False, trans_x2
    )
    actual_output, actual_gather_out = test_utils.convert_ms_tensors_to_numpy_arrays(
        (actual_output_tensor, actual_gather_out_tensor)
    )
    np.testing.assert_allclose(actual_output, expected_output, 0, 0)
    if gather_output:
        np.testing.assert_allclose(actual_gather_out, expected_gather_out, 0, 0)


@ops_binary_cases.ops_binary_cases(ops_binary_cases.OpsBinaryCase(
    input_info=[((128, 256), np.float16), ((256, 512), np.float16)],
    output_info=[((1024, 512), np.float16), ((1024, 256), np.float16)],
    is_parallel=True,
))
def ops_all_gather_matmul_binary_case1(
        input_binary_data: list_annotation[np.ndarray] = None, output_binary_data: list_annotation[np.ndarray] = None
    ) -> None:
    binary_compare(input_binary_data, output_binary_data, False, True)


@ops_binary_cases.ops_binary_cases(ops_binary_cases.OpsBinaryCase(
    input_info=[((128, 256), np.float16), ((512, 256), np.float16)],
    output_info=[((1024, 512), np.float16), ((1024, 256), np.float16)],
    is_parallel=True,
))
def ops_all_gather_matmul_binary_case2(
        input_binary_data: list_annotation[np.ndarray] = None, output_binary_data: list_annotation[np.ndarray] = None
    ) -> None:
    binary_compare(input_binary_data, output_binary_data, True, True)


@ops_binary_cases.ops_binary_cases(ops_binary_cases.OpsBinaryCase(
    input_info=[((128, 256), np.float16), ((256, 512), np.float16)],
    output_info=[((1024, 512), np.float16), ((0,), np.float16)],
    is_parallel=True,
))
def ops_all_gather_matmul_binary_case3(
        input_binary_data: list_annotation[np.ndarray] = None, output_binary_data: list_annotation[np.ndarray] = None
    ) -> None:
    binary_compare(input_binary_data, output_binary_data, False, False)


@ops_binary_cases.ops_binary_cases(ops_binary_cases.OpsBinaryCase(
    input_info=[((128, 256), np.float32), ((256, 512), np.float32)],
    output_info=[((1024, 512), np.float32), ((1024, 256), np.float32)],
    is_parallel=True,
))
def ops_all_gather_matmul_binary_case4(
        input_binary_data: list_annotation[np.ndarray] = None, output_binary_data: list_annotation[np.ndarray] = None
    ) -> None:
    binary_compare(input_binary_data, output_binary_data, False, True)


@pytest.mark.parametrize('mode, jit_level', [
    (ms.GRAPH_MODE, 'O0'),
    (ms.GRAPH_MODE, 'O1'),
    (ms.GRAPH_MODE, 'O2'),
    (ms.PYNATIVE_MODE, ''),
])
def test_all_gather_matmul_binary_cases(mode: int, jit_level: str) -> None:
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
                 torch_npu.npu_all_gather_base_mm forword calculation.
    """
    ms.communication.init()
    ms.set_context(mode=mode, device_target='Ascend')
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': jit_level})

    ops_all_gather_matmul_binary_case1()
    ops_all_gather_matmul_binary_case2()
    if jit_level != 'O2':
        ops_all_gather_matmul_binary_case3()
    ops_all_gather_matmul_binary_case4()
