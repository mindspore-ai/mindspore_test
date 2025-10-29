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
"""Utility helpers for operation testing.

This module provides:
- OpInfo dataclass describing operator metadata for tests.
- BinaryOpInfo convenience subclass with defaults for binary ops.
- sample_inputs_binary_op_func: canonical sample input generator for binary ops.
"""
import functools
from typing import Callable, Optional
from dataclasses import dataclass, field
from tests.st.ops.share._op_info.op_common import (
    MEDIUM_DIM_SIZE, SMALL_DIM_SIZE, EXTRA_SMALL_DIM_SIZE,
    get_default_loss
)
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor

@dataclass
class OpInfo:
    """Metadata describing an operator under test.

    Attributes:
        name: Short op alias used in logs and test names.
        op: MindSpore callable implementation.
        op_func_grad: MindSpore callable used for gradient nets; falls back to
            ``op`` if not provided (e.g., when kwargs need special handling).
        ref: Reference implementation (e.g., PyTorch/NumPy callable).
        tensor_variant: Tensor method variant of the operator, if applicable.

        dtypes_ascend: Supported MindSpore dtypes on Ascend devices.
        dtypes_ascend910b: Supported dtypes specifically on Ascend 910B.
        dtypes_cpu: Supported dtypes on CPU.
        dtypes_gpu: Supported dtypes on GPU.
        dtypes_intersection: Intersection of supported dtypes across all listed
            backends. Auto-populated in ``__post_init__`` if left empty.

        op_sample_inputs_func: Function that generates sample inputs for tests.
        op_error_inputs_func: Function that generates error/negative samples.
        op_dynamic_inputs_func: Function that generates dynamic-shape samples.

        is_differentiable: Whether gradients are expected/computed for the op.
        is_inplace_op: Whether the op mutates its input (in-place semantics).
        convert_extra_uint: Whether to convert extra uint dtypes for references
            that do not support them (e.g., PyTorch).
        convert_half_to_float: Whether to cast float16 to float32 for reference
            computation on backends where half precision is not supported.

        compare_method: Comparison strategy, e.g. 'default_golden',
            'single_golden', or 'double_golden'.
        default_golden_loss_func: Callable returning default numeric tolerance
            (rtol/atol) based on dtype.
    """
    # name of primitive, defined in xxx_op.yaml file.
    name: str
    op: Optional[Callable] = None
    op_func_grad: Optional[Callable] = None
    ref: Optional[Callable] = None
    tensor_variant: Optional[Callable] = None

    # dtypes supported by each backend
    dtypes_ascend: tuple = field(default_factory=tuple)
    dtypes_ascend910b: tuple = field(default_factory=tuple)
    dtypes_cpu: tuple = field(default_factory=tuple)
    dtypes_gpu: tuple = field(default_factory=tuple)
    dtypes_intersection: tuple = field(default_factory=tuple)

    # function to generate sample inputs for the op.
    op_sample_inputs_func: Optional[Callable] = None
    # function to generate error inputs for the op.
    op_error_inputs_func: Optional[Callable] = None
    # function to generate dynamic inputs for the op.
    op_dynamic_inputs_func: Optional[Callable] = None

    # extra options for the op.
    is_differentiable: Optional[bool] = True
    is_inplace_op: Optional[bool] = False
    convert_extra_uint: Optional[bool] = True
    convert_half_to_float: Optional[bool] = False

    # comparison params
    compare_method: Optional[str] = 'default_golden'
    default_golden_loss_func: Optional[Callable] = get_default_loss

    def __post_init__(self):
        if not self.dtypes_intersection:
            self.dtypes_intersection = tuple(
                set(self.dtypes_ascend) & set(self.dtypes_ascend910b) & set(self.dtypes_cpu) & set(self.dtypes_gpu)
            )
        if self.op_func_grad is None:
            self.op_func_grad = self.op


def sample_inputs_binary_op_func(op_info: OpInfo, dtype, device=None, **kwargs):
    """Yield shape/broadcasting cases for binary ops.

    Generates a variety of tensor shape combinations, including scalars,
    vectors, broadcasting pairs, and empty-dimension cases.
    """
    XS = EXTRA_SMALL_DIM_SIZE
    S = EXTRA_SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE

    make_func = functools.partial(
        make_tensor,
        device=device,
        dtype=dtype,
    )

    shapes = (
        ((), ()),
        ((S,), ()),
        ((S, 1), (S,)),
        ((M, S), ()),
        ((S, M, S), (M, S)),
        ((S, M, S), (S, M, S)),
        ((M, 1, S), (M, S)),
        ((M, 1, S), (1, M, S)),
        ((0, 1, XS), (0, M, XS)),
    )

    for input_shape, other_shape in shapes:
        _input = make_func(input_shape)
        _other = make_func(other_shape)

        yield OpSampleInput(
            _input,
            op_args=(_other,),
            op_name=op_info.name,
        )


class BinaryOpInfo(OpInfo):

    def __init__(
            self,
            name: str,
            *,
            op_sample_inputs_func: Optional[Callable] = sample_inputs_binary_op_func,
            **kwargs,
    ):
        super().__init__(
            name,
            op_sample_inputs_func=op_sample_inputs_func,
            **kwargs,
        )
