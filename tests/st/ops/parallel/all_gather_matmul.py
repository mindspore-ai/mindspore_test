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

import mindspore as ms
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools import ops_binary_cases
from tests.st.utils import test_utils

if sys.version_info >= (3, 9):
    list_annotation, tuple_annotation = list, tuple
else:
    from typing import List, Tuple
    list_annotation, tuple_annotation = List, Tuple


@test_utils.run_with_cell
def small_ops_func(input_: ms.Tensor, x2: ms.Tensor) -> tuple_annotation[ms.Tensor]:
    gather_out = ms.ops.AllGather()(input_)
    output = ms.ops.MatMul()(gather_out, x2)
    return output, gather_out


@test_utils.run_with_cell
def fusion_ops_func(
        input_: ms.Tensor,
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
        input_,
        x2,
        group,
        world_size,
        bias=bias,
        gather_index=gather_index,
        gather_output=gather_output,
        comm_turn=comm_turn,
        trans_input=trans_input,
        trans_x2=trans_x2,
    )


@ops_binary_cases.ops_binary_cases(ops_binary_cases.OpsBinaryCase(
    input_info=[((1, 256), np.float16), ((256, 1), np.float16)],
    output_info=[((2, 1), np.float16), ((2, 256), np.float16)],
    is_parallel=True,
))
def ops_all_gather_matmul_binary_case(
        input_binary_data: list_annotation[np.ndarray] = None, output_binary_data: list_annotation[np.ndarray] = None
    ) -> None:
    input_, x2 = test_utils.get_inputs_tensor(input_binary_data)
    if input_.dtype == ms.float32:
        input_ = input_.astype(ms.bfloat16)
        x2 = x2.astype(ms.bfloat16)
    expected_output, expected_gather_out = output_binary_data
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    world_size = ms.communication.get_group_size()
    actual_output_tensor, actual_gather_out_tensor = fusion_ops_func(
        input_, x2, group, world_size, None, 0, True, 0, False, False
    )
    actual_output, actual_gather_out = test_utils.convert_ms_tensors_to_numpy_arrays(
        (actual_output_tensor, actual_gather_out_tensor)
    )
    np.testing.assert_allclose(actual_output, expected_output, 0, 0)
    np.testing.assert_allclose(actual_gather_out, expected_gather_out, 0, 0)


def test_binary_case(_init):
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
        torch_npu.npu_all_gather_base_mm forward calculation.
    """
    ops_all_gather_matmul_binary_case()


def get_dynamic_func(group: str, world_size: int) \
    -> Callable[[ms.Tensor, ms.Tensor, Optional[ms.Tensor], int, bool, int, bool, bool], tuple_annotation[ms.Tensor]]:
    def func(
            input_: ms.Tensor,
            x2: ms.Tensor,
            bias: Optional[ms.Tensor],
            gather_index: int,
            gather_output: bool,
            comm_turn: int,
            trans_input: bool,
            trans_x2: bool,
        ) -> tuple_annotation[ms.Tensor]:
        return ms.ops.all_gather_matmul(
            input_,
            x2,
            group,
            world_size,
            bias=bias,
            gather_index=gather_index,
            gather_output=gather_output,
            comm_turn=comm_turn,
            trans_input=trans_input,
            trans_x2=trans_x2,
        )
    return func


def test_dynamic_shape(_init):
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the dynamic shape function of forward calculation.
    Expectation: The result of forward calculation with inputs in dynamic shapes is equal to the result of forward
        calculation with inputs in static shapes.
    """
    rank = ms.communication.get_rank()
    np.random.seed(rank)
    first_input = test_utils.generate_random_tensor((1, 256), ms.float16)
    first_x2 = test_utils.generate_random_tensor((256, 1), ms.float16)
    second_input = test_utils.generate_random_tensor((2, 256), ms.float16)
    second_x2 = test_utils.generate_random_tensor((1, 256), ms.float16)
    inputs_seq = [
        [first_input, first_x2, None, 0, True, 0, False, False],
        [second_input, second_x2, None, 0, False, 0, False, True],
    ]

    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    world_size = ms.communication.get_group_size()
    dynamic_func = get_dynamic_func(group, world_size)
    TEST_OP(
        dynamic_func,
        inputs_seq,
        disable_generalize_test=True,
        disable_mode=['GRAPH_MODE_GE'],
        case_config={'disable_input_check': True, 'disable_grad': True},
    )


def test_precision_with_ms_small_ops(_init):
    """
    Feature: mindspore.ops.all_gather_matmul
    Description: Test the precision of forward calculation.
    Expectation: The result of mindspore.ops.all_gather_matmul forward calculation is equal to the result of
        mindspore.ops.AllGahter and mindspore.ops.MatMul forward calculation.
    """
    rank = ms.communication.get_rank()
    np.random.seed(rank)
    input_ = test_utils.generate_random_tensor((1, 256), ms.float16)
    x2 = test_utils.generate_random_tensor((256, 1), ms.float16)

    expected_output, expected_gather_out = small_ops_func(input_, x2)
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    world_size = ms.communication.get_group_size()
    actual_output, actual_gather_out = fusion_ops_func(
        input_, x2, group, world_size, None, 0, True, 0, False, False
    )
    assert actual_output.isclose(expected_output, rtol=1e-03, atol=1e-03).all()
    assert actual_gather_out.isclose(expected_gather_out, rtol=1e-03, atol=1e-03).all()
