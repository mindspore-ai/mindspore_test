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
import itertools
import math
import numpy as np
import mindspore as ms
from typing import Callable, Optional
from dataclasses import dataclass, field
from tests.st.ops.share._op_info.op_common import (
    MEDIUM_DIM_SIZE, SMALL_DIM_SIZE, EXTRA_SMALL_DIM_SIZE,
    dtypes_integral, dtypes_extra_uint,
    get_default_loss,
)
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, make_tensor_with_np_array
)

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
        op_reference_inputs_func: Function that generates comprehensive sample inputs for tests.
        op_dynamic_inputs_func: Function that generates dynamic-shape samples.
        op_error_inputs_func: Function that generates error/negative samples.

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

    # function to generate sample inputs (basic shape/broadcasting cases) for the op.
    op_sample_inputs_func: Optional[Callable] = None
    # function to generate reference inputs (discontinuous/special values/large or small values cases) for the op.
    op_reference_inputs_func: Optional[Callable] = None
    # function to generate dynamic inputs for the op.
    op_dynamic_inputs_func: Optional[Callable] = None
    # function to generate error inputs for the op.
    op_error_inputs_func: Optional[Callable] = None

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

# op_sample_inputs_func for ops
def sample_inputs_binary_op_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Yield shape/broadcasting cases for binary ops.

    Generates a variety of tensor shape combinations, including scalars,
    vectors, broadcasting pairs, and empty-dimension cases.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
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

# op_reference_inputs_func for ops
def _generate_binary_op_tensors_sample_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor-tensor sample inputs for a binary op.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering empty/scalar/various shapes.
    """
    shapes = (
        (0,),        # empty tensors
        (1, 0, 3),   # empty tensors
        (),          # scalar tensors
        (20,),       # 1D tensors with small size
        (812,),      # 1D tensors with medium size
        (1029, 917), # 2D tensors with large size
    )

    for shape in shapes:
        yield OpSampleInput(
            op_input=make_tensor(shape, dtype, device=device),
            op_args=(make_tensor(shape, dtype, device=device),),
            op_kwargs={},
            op_name=f"{op_info.name}_tensor_inputs",
        )


def _generate_binary_op_small_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor-tensor inputs with small values for edge cases.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering representative small float/int/unsigned values.
    """
    _uint_values = (0, 1, 31, 63, 128, 195, 254)
    _int_values = (0, -1, 1, -63, 63, -127, 127, -128)
    _float_vals = (
        0.0,
        -0.0,
        -1.0,
        1.0,
        -1.25e-1,
        1.25e-1,
        -1e-3,
        1e-3,
        -math.pi / 2,
        math.pi / 2,
        -math.pi + 1e-5,
        math.pi - 1e-5,
        -math.pi,
        math.pi,
        -math.pi - 1e-5,
        math.pi + 1e-5,
    )

    _input_values = []
    _other_values = []

    if dtype.is_floating_point:
        values_group = itertools.product(_float_vals, _float_vals)
    elif dtype.is_complex:
        _float_values_group = itertools.product(_float_vals, _float_vals)
        _complex_values = [complex(r, i) for r, i in _float_values_group]
        values_group = itertools.product(_complex_values, _complex_values)
    elif dtype in dtypes_integral and dtype != ms.bool_:
        if dtype in dtypes_extra_uint or dtype == ms.uint8:
            values_group = itertools.product(_uint_values, _uint_values)
        else:
            values_group = itertools.product(_int_values, _int_values)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}!")

    for l_value, r_value in values_group:
        _input_values.append(l_value)
        _other_values.append(r_value)

    yield OpSampleInput(
        op_input= make_tensor_with_np_array(np.array(_input_values), dtype=dtype, device=device),
        op_args=(make_tensor_with_np_array(np.array(_other_values), dtype=dtype, device=device),),
        op_kwargs={},
        op_name=f"{op_info.name}_small_value_tensor_inputs",
    )


def _generate_binary_op_large_value_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate tensor-tensor inputs with large values for stress testing.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering large-magnitude float/complex/integer values.
    """
    _int_values = (-1023, 1023, -10922, 10922)
    _fp16_values = (-521, 521, -996.9, 996.9, -13141.5, 13141.5)
    _fp32_values = _fp16_values + (-5062757.7, 5062757.7, -1e20, 1e20)

    _input_values = []
    _other_values = []

    if dtype == ms.float16:
        values_group = itertools.product(_fp16_values, _fp16_values)
    elif dtype.is_floating_point:
        values_group = itertools.product(_fp32_values, _fp32_values)
    elif dtype.is_complex:
        _float_values_group = itertools.product(_fp32_values, _fp32_values)
        _complex_values = [complex(r, i) for r, i in _float_values_group]
        values_group = itertools.product(_complex_values, _complex_values)
    elif dtype in (ms.int16, ms.int32, ms.int64):
        values_group = itertools.product(_int_values, _int_values)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}!")

    for l_value, r_value in values_group:
        _input_values.append(l_value)
        _other_values.append(r_value)

    yield OpSampleInput(
        op_input= make_tensor_with_np_array(np.array(_input_values), dtype=dtype, device=device),
        op_args=(make_tensor_with_np_array(np.array(_other_values), dtype=dtype, device=device),),
        op_kwargs={},
        op_name=f"{op_info.name}_large_value_tensor_inputs",
    )

def _generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate broadcasting and (dis)contiguous tensor inputs.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Inputs covering broadcasting and contiguous/non-contiguous memory layouts.
    """

    shapes = (
        ((1,), ()),
        ((3,), ()),
        ((1,), (3,)),
        ((3, 1), (3,)),
        ((1, 3), (3,)),
        ((3, 2), (2,)),
        ((1, 2, 3), (3,)),
        ((1, 2, 3), (2, 3)),
        ((2, 1, 3), (2, 3)),
        ((2, 3, 4), ()),
        ((2, 1, 2), (1, 3, 2)),
    )

    for shape_pair, discontiguous in itertools.product(shapes, [False, True]):
        _input_shape, _other_shape = shape_pair

        yield OpSampleInput(
            op_input=make_tensor(_input_shape, dtype=dtype, device=device, discontiguous=discontiguous),
            op_args=(make_tensor(_other_shape, dtype=dtype, device=device, discontiguous=discontiguous),),
            op_kwargs={},
            op_name=f"{op_info.name}_broadcasting_and_discontiguous_tensor_inputs",
        )


def _generate_binary_op_scalar_inputs_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate scalar-tensor/tensor-scalar/scalar-scalar inputs per support flags.

    Args:
        op_info: Operator metadata (includes Python scalar support flags).
        dtype: Data type for generated tensors.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Sample inputs mixing scalars and tensors.
    """

    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    shapes = (
        (),
        (2,),
        (0, 2),
    )
    _scalar = 1.0 if dtype.is_complex else make_func(()).item()

    if op_info.supports_left_python_scalar:
        for shape in shapes:
            yield OpSampleInput(
                op_input=make_func(shape),
                op_args=(_scalar,),
                op_kwargs={},
                op_name=f"{op_info.name}_scalarxtensor_inputs",
            )

    if op_info.supports_right_python_scalar:
        for shape in shapes:
            yield OpSampleInput(
                op_input=_scalar,
                op_args=(make_func(shape),),
                op_kwargs={},
                op_name=f"{op_info.name}_tensorxscalar_inputs",
            )

    if op_info.supports_both_python_scalar:
        yield OpSampleInput(
            op_input=_scalar,
            op_args=(_scalar,),
            op_kwargs={},
            op_name=f"{op_info.name}_scalarxscalar_inputs",
        )


def reference_inputs_binary_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate comprehensive reference inputs for a binary op.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Additional options (unused).

    Returns:
        Generator[OpSampleInput]: Aggregated samples from multiple input generators.
    """
    if dtype in dtypes_extra_uint:
        return
    # tensors with many kinds of shapes
    yield from _generate_binary_op_tensors_sample_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with small value
    if dtype != ms.bool_:
        yield from _generate_binary_op_small_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # tensors with large value
    if dtype not in (ms.bool_, ms.uint8, ms.int8):
        yield from _generate_binary_op_large_value_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # broadcasting tensors and contiguous or discontiguous tensors
    yield from _generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func(op_info, dtype, device, **kwargs)
    # scalarxtensor, tensorxscalar and scalarxscalar
    yield from _generate_binary_op_scalar_inputs_func(op_info, dtype, device, **kwargs)


# op_dynamic_inputs_func for ops
def dynamic_inputs_binary_op_common_func(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    """Generate dynamic-shape/rank inputs for a binary op.

    Args:
        op_info: Operator metadata.
        dtype: Data type of tensors to generate.
        device: Target device.
        kwargs: Flags such as only_dynamic_shape/only_dynamic_rank.

    Returns:
        Generator[OpDynamicInput]: Dynamic compile-time and runtime inputs.
    """
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # dyncamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None), dtype=dtype),),
                op_kwargs={},
                op_name=f"{op_info.name}_dynamic_shape_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(make_func(shape=(2, 3)),),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(4, 5)),
                    op_args=(make_func(shape=(4, 5)),),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input",
                ),
            ),
        )
    if not kwargs.get("only_dynamic_shape", False):
        # dyncamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={},
                op_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(make_func(shape=(2, 3)),),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3, 4)),
                    op_args=(make_func(shape=(2, 3, 4)),),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )


# op_error_inputs_func for ops
def error_inputs_binary_op_common_func(op_error_inputs_func = None):
    """Wrap or synthesize error-case inputs for a binary op.

    Args:
        op_error_inputs_func: Optional downstream provider to extend/override
            default error cases.

    Returns:
        Callable[..., Generator[OpErrorInput, None, None]]: A generator factory
            that yields error samples and expected exception types.
    """
    def _error_inputs_binary_op_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
        if op_error_inputs_func is not None:
            yield from op_error_inputs_func(op_info, dtype, device)

        S = SMALL_DIM_SIZE
        if not op_info.supports_left_python_scalar:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=2,
                    op_args=(make_tensor(shape=(S,), dtype=ms.int64, device=device),),
                    op_kwargs={},
                    op_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError, RuntimeError),
                op_error_info=f'left_python_scalar is not supported for {op_info.name}',
            )
        if not op_info.supports_right_python_scalar:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=make_tensor(shape=(S,), dtype=ms.int64, device=device),
                    op_args=(2,),
                    op_kwargs={},
                    op_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError, RuntimeError),
                op_error_info=f'right_python_scalar is not supported for {op_info.name}',
            )
        if not op_info.supports_both_python_scalar:
            yield OpErrorInput(
                op_sample_input=OpSampleInput(
                    op_input=1,
                    op_args=(2,),
                    op_kwargs={},
                    op_name=op_info.name,
                ),
                op_error_type=(TypeError, ValueError, RuntimeError),
                op_error_info=f'both_python_scalar is not supported for {op_info.name}',
            )

    return _error_inputs_binary_op_func


class BinaryOpInfo(OpInfo):
    """Operator meta information for binary operations.
    These operations have two tensors as input and one tensor as output usually.
    And they may have the following characteristics:
      - they are elementwise operations
      - the output shape is determined by the broadcasted input shapes usually.
      - they may have keyword arguments.
    
    Extra attributes:
      - support_tensor_type_promotion: Whether to support tensors' type promotion.
      - supports_left_python_scalar: Whether to support left python scalar.
      - supports_right_python_scalar: Whether to support right python scalar.
      - supports_both_python_scalar: Whether to support both python scalar.
    """
    def __init__(
            self,
            name: str,
            *,
            op_sample_inputs_func: Optional[Callable] = sample_inputs_binary_op_func,
            op_reference_inputs_func: Optional[Callable] = reference_inputs_binary_op_common_func,
            op_dynamic_inputs_func: Optional[Callable] = dynamic_inputs_binary_op_common_func,
            op_error_inputs_func: Optional[Callable] = None,
            support_tensor_type_promotion: Optional[bool] = True,
            supports_left_python_scalar: Optional[bool] = True,
            supports_right_python_scalar: Optional[bool] = True,
            supports_both_python_scalar: Optional[bool] = True,
            **kwargs,
    ):
        super().__init__(
            name,
            op_sample_inputs_func=op_sample_inputs_func,
            op_reference_inputs_func=op_reference_inputs_func,
            op_dynamic_inputs_func=op_dynamic_inputs_func,
            **kwargs,
        )
        self.support_tensor_type_promotion = support_tensor_type_promotion
        self.supports_left_python_scalar = supports_left_python_scalar
        self.supports_right_python_scalar = supports_right_python_scalar
        self.supports_both_python_scalar = supports_both_python_scalar
        self.op_error_inputs_func = error_inputs_binary_op_common_func(op_error_inputs_func)
