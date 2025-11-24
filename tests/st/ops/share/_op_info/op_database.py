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
- Sample input builders for add/sub ops (including alpha cases).
- Dynamic-shape input builders for rank/shape dynamism.
- Gradient wrappers for ops with kwargs.
- The operator database (op_db) and get_op_info accessor.
"""
import functools
import numpy as np
from typing import Dict, Optional
import torch
import itertools
import mindspore as ms
from mindspore import mint, mutable
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo, UnaryOpInfo, ReductionOpInfo
from tests.st.ops.share._op_info.op_info import (
    basic_reference_inputs_reduction_op_common_func,
    extra_reference_inputs_reduction_op_common_func,
    basic_reference_inputs_binary_op_common_func,
    extra_reference_inputs_binary_op_common_func,
    dynamic_inputs_binary_op_common_func
)
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, OpErrorInput,
    make_tensor, skip_sample_inputs
)
from tests.st.ops.share._op_info.op_common import (
    dtypes_as_torch, dtypes_extra_uint, SMALL_DIM_SIZE, MEDIUM_DIM_SIZE, EXTRA_SMALL_DIM_SIZE, LARGE_DIM_SIZE
)


# op_basic_reference_inputs_func for ops
def basic_sample_inputs_add_sub_ext(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate sample inputs for add/sub ops including extra alpha cases.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpSampleInput objects.
    '''
    yield from basic_reference_inputs_binary_op_common_func(op_info, dtype, device, **kwargs)

    S = SMALL_DIM_SIZE
    # Adds alpha kwarg cases
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    _input = make_arg((S, S))
    _other = make_arg((S, S))
    if dtype is not ms.bool_:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': 2},
            sample_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': True},
            sample_name=op_info.name
        )

    neg_alpha = -0.1415 if (dtype.is_floating_point or dtype.is_complex) else -3
    if dtype in dtypes_extra_uint:
        neg_alpha = abs(neg_alpha)

    _input = make_arg((S, S))
    _other = make_arg((S, S))
    if dtype is not ms.bool_:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': neg_alpha},
            sample_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': False},
            sample_name=op_info.name
        )

# op_dynamic_inputs_func for ops
def dynamic_sample_inputs_add_sub_ext(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    '''
    Generate dynamic inputs for add/sub_ext ops.
    Args:
        op_info: OpInfo object.
        dtype: Data type of the tensors.
        device: Device of the tensors.
        kwargs: Additional keyword arguments.
    Returns:
        Generator of OpDynamicInput objects.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)
    if not kwargs.get("only_dynamic_rank", False):
        # add/sub_ext dynamic shape
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, None, None, None, None), dtype=dtype),
                op_args=(ms.Tensor(shape=(None, None, None, 1, None), dtype=dtype),),
                op_kwargs={"alpha": mutable(input_data=3.3, dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5, 8, 5, 4)),
                    op_args=(make_func(shape=(5, 5, 8, 1, 4)),),
                    op_kwargs={"alpha": mutable(input_data=4.3, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 8, 8, 4)),
                    op_args=(make_func(shape=(9, 9, 8, 1, 4)),),
                    op_kwargs={"alpha": mutable(input_data=-2.1, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
            )
        )
    if not kwargs.get("only_dynamic_shape", False):
        # add/sub_ext dynamic rank
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={"alpha": mutable(input_data=2.33, dynamic_len=False)},
                sample_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5)),
                    op_args=(make_func(shape=(5, 5)),),
                    op_kwargs={"alpha": mutable(input_data=9.6, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 7)),
                    op_args=(make_func(shape=(9, 9, 7)),),
                    op_kwargs={"alpha": mutable(input_data=10.10, dynamic_len=False)},
                    sample_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
            )
        )

# op_error_inputs_func for ops
def error_inputs_add_sub_ext_func(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate error inputs for add/sub_ext ops.
    '''
    # other shape does not match input
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.float32),
            op_args=(make_tensor(shape=(3,), dtype=ms.float32),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=ValueError,
        op_error_info='other shape does not match input',
    )
    # other is not tensor or number
    yield OpErrorInput(
        op_sample_input=OpSampleInput(
            op_input=make_tensor(shape=(2,), dtype=ms.float32),
            op_args=((1, 2),),
            op_kwargs={},
            sample_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='other is not tensor or number',
    )

def basic_sample_inputs_mint_repeat_interleave(
    op_info: OpInfo,
    dtype,
    device=None,
    **kwargs
):
    '''
    Generate basic sample inputs for mint.repeat_interleave.

    Cases covered (aligned with PyTorch repeat_interleave samples, adapted to MindSpore):
      - scalar tensor with repeats as Python int
      - 3D tensor with repeats as Python int (no dim / with dim)
      - 3D tensor with repeats as Tensor (per-index repeats) with dim
      - 2D tensor with repeats as Tensor and explicit output_size
    '''
    make_input = functools.partial(make_tensor, device=device, dtype=dtype)

    # () with repeats=2
    yield OpSampleInput(
        op_input=make_input(()),
        op_args=(2,),
        op_kwargs={},
        sample_name=f"{op_info.name}_scalar_repeats_int"
    )

    # (2, 3, 4) with repeats=2 (no dim)
    yield OpSampleInput(
        op_input=make_input((2, 3, 4)),
        op_args=(2,),
        op_kwargs={},
        sample_name=f"{op_info.name}_nd_repeats_int_nodim"
    )

    # (2, 3, 4) with repeats=2, dim=1
    yield OpSampleInput(
        op_input=make_input((2, 3, 4)),
        op_args=(2,),
        op_kwargs={'dim': 1},
        sample_name=f"{op_info.name}_nd_repeats_int_dim1"
    )

    # (2, 3, 4) with repeats as Tensor([0, 1, 2]), dim=1
    repeats_tensor_dim1 = ms.Tensor([0, 1, 2], dtype=ms.int32)
    yield OpSampleInput(
        op_input=make_input((2, 3, 4)),
        op_args=(repeats_tensor_dim1,),
        op_kwargs={'dim': 1},
        sample_name=f"{op_info.name}_nd_repeats_tensor_dim1"
    )

    # (4, 1) with repeats as Tensor([0,1,2,3]), dim=0, output_size=6
    repeats_tensor_dim0 = ms.Tensor([0, 1, 2, 3], dtype=ms.int32)
    yield OpSampleInput(
        op_input=make_input((4, 1)),
        op_args=(repeats_tensor_dim0,),
        op_kwargs={'dim': 0, 'output_size': 6},
        sample_name=f"{op_info.name}_nd_repeats_tensor_dim0_output_size"
    )

    # (3,) with repeats as random Tensor([...]), dim=0
    rand_repeats = ms.Tensor(np.random.randint(1, 10, (3,)), dtype=ms.int32)
    yield OpSampleInput(
        op_input=make_input((3,)),
        op_args=(rand_repeats,),
        op_kwargs={'dim': 0},
        sample_name=f"{op_info.name}_1d_rand_repeats_tensor_dim0"
    )

def basic_sample_inputs_mint_arange(
    op_info: OpInfo,
    dtype=None,
    device=None,
    **kwargs
):
    '''
    Generate basic sample inputs for mint.arange, aligned with PyTorch's sample_inputs_arange.
    Each tuple is (start, end, step), where None means "omit that positional".
    The outer dtype is forwarded via op_kwargs['dtype'] to fix output dtype (like PyTorch does).
    '''
    int_samples = (
        (-1, 2, 2),          # positive direction
        (2, -3, -1),         # negative direction
        (-3, -10, -2),       # additional negative direction with even step
        (1, 1, 1),           # start == end
        (1, 1, -1),          # start == end with negative step
        (0, -8, -4),         # divides evenly (negative)
        (1, 5, 2),           # divides evenly (positive)
        (False, True, True), # bool inputs
        (0, 1, None),        # default step
        (None, 3, None),     # default start (single-arg form)
    )

    def to_float(start, end, step):
        start = (start + 0.1) if start is not None else None
        end = end + 0.1
        step = float(step) if step is not None else None
        return start, end, step

    float_samples = (
        (0.0, -8.0 - 1e-6, -4.0),  # includes endpoint
        (1.0, 5.0 + 1e-6,  2.0),   # includes endpoint
        (0.0, -8.0,       -4.0),
        (1.0, 5.0,         2.0),
        *(to_float(s, e, t) for (s, e, t) in int_samples),
    )

    large_samples = (
        (0, 10000, None),
    )

    samples = int_samples
    # Only add float_samples when output dtype is floating-point;
    # for integer dtypes, mixing float ranges can cause length mismatches vs MindSpore behavior.
    if dtype is not None and getattr(dtype, "is_floating_point", False):
        samples += float_samples
    if dtype not in (ms.int8, ms.uint8):
        samples += large_samples

    for start, end, step in samples:
        if start is None:
            op_input = end
            op_args = ()
        else:
            op_input = start
            op_args = (end,) if step is None else (end, step)

        op_kwargs = {}
        if dtype is not None:
            op_kwargs['dtype'] = dtype

        yield OpSampleInput(
            op_input=op_input,
            op_args=op_args,
            op_kwargs=op_kwargs,
            sample_name=op_info.name,
        )

# op_func_without_kwargs, used by gradient comparison if there are kwargs in op
def add_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.add(x, y, alpha=alpha)

def sub_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.sub(x, y, alpha=alpha)

def equal_func_grad(x, other):
    return mint.equal(x, other)

def eq_func_grad(x, other):
    return mint.eq(x, other)

def greater_equal_func_grad(x, other):
    return mint.greater_equal(x, other)

def greater_func_grad(x, other):
    return mint.greater(x, other)

def less_equal_func_grad(x, other):
    return mint.less_equal(x, other)

def less_func_grad(x, other):
    return mint.less(x, other)

def ne_func_grad(x, other):
    return mint.ne(x, other)

def maximum_func_grad(op_input, other):
    return mint.maximum(op_input, other)

def minimum_func_grad(op_input, other):
    return mint.minimum(op_input, other)

def div_func_grad(op_input, other):
    return mint.div(op_input, other)

def mul_func_grad(op_input, other):
    return mint.mul(op_input, other)

def repeat_interleave_func_grad(input_x, repeats, dim=None, output_size=None):
    return mint.repeat_interleave(input_x, repeats, dim, output_size=output_size)

def pow_ext_func_grad_without_kwargs(op_input, exponent):
    return mint.pow(op_input, exponent)


def floor_divide_ext_func_grad_without_kwargs(op_input, other):
    return mint.floor_divide(op_input, other)

# wrap tensor method for tanh
def tensor_tanh_ms(op_input):
    return op_input.tanh()

def tensor_ceil_ms(op_input):
    return op_input.ceil()

def tensor_exp_ms(op_input):
    return op_input.exp()

def tensor_log_ms(op_input):
    return op_input.log()

def tensor_neg_ms(op_input):
    return op_input.neg()

def tensor_sigmoid_ms(op_input):
    return op_input.sigmoid()

def tensor_sqrt_ms(op_input):
    return op_input.sqrt()

def tensor_square_ms(op_input):
    return op_input.square()

def tensor_select_ms(op_input, dim, index):
    return op_input.select(dim, index)

def tensor_floor_ms(op_input):
    return op_input.floor()

def tensor_abs_ms(op_input):
    return op_input.abs()

def tensor_floor_divide_ms(op_input, other):
    return op_input.floor_divide(other)

def tensor_tanh_torch(op_input):
    return op_input.tanh()

def tensor_eq_ms(op_input, x):
    return op_input.eq(x)

def tensor_repeat_interleave_ms(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim, output_size=output_size)

def tensor_repeat_interleave_torch(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim=dim, output_size=output_size)

def tensor_repeat_interleave_func_grad(op_input, repeats, dim=None, output_size=None):
    return op_input.repeat_interleave(repeats, dim, output_size=output_size)

# sample inputs functions for Tensor.repeat (method)
def basic_sample_inputs_tensor_repeat(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for Tensor.repeat (method), aligned to torch's coverage:
      - scalar tensor with 1D repeat
      - 1D tensor with zero and positive repeats
      - 2D/3D tensors with per-dimension repeats
      - keep sizes length equal to input.ndim for MindSpore compatibility
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    # Scalar () with repeat(2) -> shape (2,)
    yield OpSampleInput(
        op_input=make_x(()),
        op_args=(2,),
        op_kwargs={},
        sample_name=f"{op_info.name}_scalar_repeat_2",
    )

    # Scalar () with repeat(0, 0) -> shape (0, 0)
    yield OpSampleInput(
        op_input=make_x(()),
        op_args=(0, 0),
        op_kwargs={},
        sample_name=f"{op_info.name}_scalar_repeat_0x0",
    )

    # 1D (S,) with repeat(0) -> empty
    yield OpSampleInput(
        op_input=make_x((S,)),
        op_args=(0,),
        op_kwargs={},
        sample_name=f"{op_info.name}_1d_repeat_0",
    )

    # 1D (S,) with repeat(3) -> shape (S*3,)
    yield OpSampleInput(
        op_input=make_x((S,)),
        op_args=(3,),
        op_kwargs={},
        sample_name=f"{op_info.name}_1d_repeat_3",
    )

    # 2D (S, S) with repeat(1, 2) -> (S, 2S)
    yield OpSampleInput(
        op_input=make_x((S, S)),
        op_args=(1, 2),
        op_kwargs={},
        sample_name=f"{op_info.name}_2d_repeat_1x2",
    )

    # 2D (S, S) with repeat(2, 1) -> (2S, S)
    yield OpSampleInput(
        op_input=make_x((S, S)),
        op_args=(2, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_2d_repeat_2x1",
    )

    # 3D (2, 1, S) with repeat(2, 3, 1) -> (4, 3, S)
    yield OpSampleInput(
        op_input=make_x((2, 1, S)),
        op_args=(2, 3, 1),
        op_kwargs={},
        sample_name=f"{op_info.name}_3d_repeat_2x3x1",
    )

def tensor_repeat_ms(op_input, *sizes):
    return op_input.repeat(*sizes)

def tensor_repeat_torch(op_input, *sizes):
    return op_input.repeat(*sizes)

def tensor_eq_torch(op_input, x):
    return op_input.eq(x)

def tensor_greater_equal_ms(op_input, x):
    return op_input.greater_equal(x)

def tensor_greater_equal_torch(op_input, x):
    return op_input.greater_equal(x)

def tensor_greater_ms(op_input, x):
    return op_input.greater(x)

def tensor_greater_torch(op_input, x):
    return op_input.greater(x)

def tensor_less_equal_ms(op_input, x):
    return op_input.less_equal(x)

def tensor_less_equal_torch(op_input, x):
    return op_input.less_equal(x)

def tensor_less_ms(op_input, x):
    return op_input.less(x)

def tensor_less_torch(op_input, x):
    return op_input.less(x)

def tensor_ne_ms(op_input, x):
    return op_input.ne(x)

def tensor_ne_torch(op_input, x):
    return op_input.ne(x)

def tensor_gt_ms(op_input, x):
    return op_input.gt(x)

def tensor_gt_torch(op_input, x):
    return op_input.gt(x)

def tensor_le_ms(op_input, x):
    return op_input.le(x)

def tensor_le_torch(op_input, x):
    return op_input.le(x)

def tensor_lt_ms(op_input, x):
    return op_input.lt(x)

def tensor_lt_torch(op_input, x):
    return op_input.lt(x)

def tensor_maximum_ms(op_input, other):
    return op_input.maximum(other)

def tensor_maximum_torch(op_input, other):
    return op_input.maximum(other)

def tensor_minimum_ms(op_input, other):
    return op_input.minimum(other)

def tensor_minimum_torch(op_input, other):
    return op_input.minimum(other)

def tensor_mul_ms(op_input, other):
    return op_input.mul(other)

def tensor_mul_torch(op_input, other):
    return op_input.mul(other)

def tensor_ceil_torch(op_input):
    return op_input.ceil()

def tensor_exp_torch(op_input):
    return op_input.exp()

def tensor_log_torch(op_input):
    return op_input.log()

def tensor_neg_torch(op_input):
    return op_input.neg()

def tensor_select_torch(op_input, dim, index):
    return op_input.select(dim, index)

def tensor_sigmoid_torch(op_input):
    return op_input.sigmoid()

def tensor_sqrt_torch(op_input):
    return op_input.sqrt()

def tensor_square_torch(op_input):
    return op_input.square()

def tensor_floor_torch(op_input):
    return op_input.floor()

def tensor_abs_torch(op_input):
    return op_input.abs()

def tensor_floor_divide_torch(op_input, other):
    return op_input.floor_divide(other)

# wrap nn method for tanh
def nn_tanh_ms(op_input):
    return mint.nn.Tanh()(op_input)

def nn_tanh_torch(op_input):
    return torch.nn.Tanh()(op_input)

# sample inputs functions for chunk
def basic_sample_inputs_mint_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.chunk op.
    '''
    S = SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((S, S, S), (2,)),     # chunks only
        ((S, S, S), (S, 1)),   # chunks=S, dim=1
        ((S, S, S), (S, -1)),  # chunks=S, dim=-1
    )
    for shape, args in cases:
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=args,
            op_kwargs={},
            sample_name=op_info.name,
        )

def extra_sample_inputs_mint_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.chunk op.
    '''
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)

    # 0D is not supported by mint.chunk, will be added to error cases later
    cases = (
        ((3,), 5, 0),                      # 1D: ragged last (chunks > size), dim=0
        ((3, 6), 3, 1),                    # 2D: equal split, dim=1
        ((2, 3, 4), 3, -1),                # 3D: non-equal split, last dim
        ((2, 2, 2, 2), 2, 0),              # 4D: equal split, dim=0
        ((2, 2, 2, 3, 2), 2, 3),           # 5D: non-equal split, dim=3
        ((2, 2, 2, 2, 2, 3), 4, 5),        # 6D: ragged last (chunks > size), dim=5
        ((2, 2, 2, 2, 2, 2, 3), 2, -1),    # 7D: non-equal split, last dim
        ((2, 2, 2, 2, 2, 2, 2, 4), 2, 7),  # 8D: equal split, dim=7
    )
    for shape, chunks, dim in cases:
        yield OpSampleInput(
            op_input=make_arg(shape),
            op_args=(chunks, dim),
            op_kwargs={},
            sample_name=op_info.name,
        )

def dynamic_sample_inputs_mint_chunk(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic sample inputs for mint.chunk op.
    '''
    # chunk infer does NOT support dynamic rank and does NOT support the case
    # when the dimension specified by `dim` is dynamic. So we ensure the `dim`
    # dimension size is static at compile time, while other dimensions can be dynamic.
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Case A: 2D input, dim=1 static, other dim dynamic
        chunks, dim = 4, 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(None, 6), dtype=dtype),  # dim=1 is static 6
                op_args=(chunks, dim),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 6)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Case B: 3D input, dim=0 static, other dims dynamic
        chunks, dim = 3, 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim=0 is static 6
                op_args=(chunks, dim),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

# sample inputs functions for gather
def basic_sample_inputs_mint_gather(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.gather op.
    Cover 1D/2D common cases.
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    # index helper: default int64, low=0; call with shape=..., high=extent
    make_index = functools.partial(make_tensor, device=device, low=0, dtype=ms.int64)

    # 1D: dim=0, index length <= extent
    x_shape = (S,)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(S,), high=x_shape[0])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 2D: dim=0 and dim=1
    x_shape = (S, S)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(S, S), high=x_shape[0])),
        op_kwargs={},
        sample_name=op_info.name,
    )
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(1, make_index(shape=(S, S // 2), high=x_shape[1])),
        op_kwargs={},
        sample_name=op_info.name,
    )


def extra_sample_inputs_mint_gather(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate extra sample inputs for mint.gather op.
    Requirements:
    - cover 0D and 3D..8D (1D/2D already included in basic).
    - include an empty index case.
    Keep shapes small for resource efficiency.
    '''
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)
    make_index = functools.partial(make_tensor, device=device, low=0, dtype=ms.int64)

    # 0D scalar input, dim=0, index: scalar 0
    x_shape = ()
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(), high=1)),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # Empty index tensor case (1D input). Although 1D was in basic, this is a distinct edge case.
    x_shape = (S,)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(0,), high=1, dtype=ms.int32)),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 3D: gather along middle dim (dim=1)
    x_shape = (2, 3, 4)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(1, make_index(shape=(2, 2, 4), high=x_shape[1])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 4D: negative dim (-1)
    x_shape = (2, 2, 3, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(-1, make_index(shape=(2, 2, 3, 1), high=x_shape[-1])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 5D: dim=3, non-dim axes of index <= input
    x_shape = (2, 2, 2, 3, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(3, make_index(shape=(2, 2, 2, 2, 2), high=x_shape[3])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 6D: dim=0
    x_shape = (3, 2, 2, 2, 2, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(2, 2, 2, 2, 2, 2), high=x_shape[0])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 7D: last dim
    x_shape = (2, 2, 2, 2, 2, 2, 3)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(-1, make_index(shape=(2, 2, 2, 2, 2, 2, 2), high=x_shape[-1])),
        op_kwargs={},
        sample_name=op_info.name,
    )

    # 8D: dim=5
    x_shape = (2, 2, 2, 2, 2, 4, 2, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(5, make_index(shape=(2, 2, 2, 2, 2, 2, 2, 2), high=x_shape[5])),
        op_kwargs={},
        sample_name=op_info.name,
    )


def dynamic_sample_inputs_mint_gather(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic sample inputs for mint.gather op.
    Consider both dynamic_shape and dynamic_rank.
    Notes per infer logic (gather_d.cc):
      - dim should be a scalar constant.
      - index.rank must equal input.rank.
      - For non-dim axes, dynamic shapes lead to retry; so keep non-dim axes static at compile time.
    '''
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 2D input, dim=1 static length, axis 0 static at compile time
        dim = 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(5, None), dtype=dtype),  # non-dim axis (0) static
                op_args=(dim, ms.Tensor(shape=(5, None), dtype=ms.int64)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(dim, make_tensor(shape=(5, 3), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8)),
                    op_args=(dim, make_tensor(shape=(5, 4), dtype=ms.int64, device=device, low=0, high=8)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Dynamic shape case 2: 3D input, dim=0 (static), other axes dynamic only on input
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim axis static
                op_args=(dim, ms.Tensor(shape=(None, 2, 2), dtype=ms.int64)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(dim, make_tensor(shape=(3, 2, 2), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(dim, make_tensor(shape=(4, 2, 2), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(dim, ms.Tensor(shape=None, dtype=ms.int64)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(dim, make_tensor(shape=(2,), dtype=ms.int64, device=device, low=0, high=3)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(dim, make_tensor(shape=(2, 3), dtype=ms.int64, device=device, low=0, high=2)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )

# sample inputs functions for mint.nn.functional.interpolate
def _normalize_mode_and_ranks(mode: str):
    # Map the op_db mode to runtime interpolate mode and supported ranks
    if mode in ("nearest1d", "nearest2d", "nearest3d"):
        internal_mode = "nearest"
        ranks = {"nearest1d": [1], "nearest2d": [2], "nearest3d": [3]}[mode]
    else:
        internal_mode = mode
        ranks = {
            "nearest": [1, 2, 3],
            "linear": [1],
            "bilinear": [2],
            "bicubic": [2],
            "trilinear": [3],
        }[mode]
    return internal_mode, ranks


def basic_sample_inputs_mint_interpolate(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate basic sample inputs for mint.nn.functional.interpolate.
    Reference torch's sample_inputs_interpolate:
      - cover size and scale_factor usages
      - align_corners for linear-family: True/False/None; for nearest: None only
    '''
    mode = kwargs.get("mode")
    internal_mode, ranks = _normalize_mode_and_ranks(mode)

    if internal_mode in ("linear", "bilinear", "bicubic", "trilinear"):
        align_corners_options = (True, False, None)
    else:
        align_corners_options = (None,)

    N, C = 1, 1
    D = 2
    S_small = 2
    S_large = 3

    def shape_with_nc(side: int, rank: int):
        return tuple([N, C] + [side] * rank)

    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, low=-1, high=1)

    for align_corners in align_corners_options:
        for rank in ranks:
            # Using size
            size_small = tuple([S_small] * rank)
            size_large = tuple([S_large] * rank)
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": size_small,
                    "scale_factor": None,
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": None,
                },
                sample_name=op_info.name,
            )
            yield OpSampleInput(
                op_input=make_arg(shape_with_nc(D, rank)),
                op_args=(),
                op_kwargs={
                    "size": size_large,
                    "scale_factor": None,
                    "mode": internal_mode,
                    "align_corners": align_corners,
                    "recompute_scale_factor": None,
                },
                sample_name=op_info.name,
            )

            # Using scale_factor and varying recompute_scale_factor
            for recompute in (False, True):
                for scale in (1.7, 0.6):
                    yield OpSampleInput(
                        op_input=make_arg(shape_with_nc(D, rank)),
                        op_args=(),
                        op_kwargs={
                            "size": None,
                            "scale_factor": scale,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": recompute,
                        },
                        sample_name=op_info.name,
                    )


def dynamic_sample_inputs_mint_interpolate(op_info: OpInfo, dtype=None, device=None, **kwargs):
    '''
    Generate dynamic sample inputs for mint.nn.functional.interpolate.
    Guideline based on upsample_forward_base inference:
      - exactly one of size or scale_factor must be provided
      - dynamic shape: spatial dims can be None at compile time
      - dynamic rank: compile with shape=None and run with specific ranks
    '''
    mode = kwargs.get("mode")
    internal_mode, ranks = _normalize_mode_and_ranks(mode)

    if internal_mode in ("linear", "bilinear", "bicubic", "trilinear"):
        dyn_align_corners = (True, False)
    else:
        dyn_align_corners = (None,)

    N, C = 1, 1
    D1 = 2
    D2 = 3
    S_target = 2

    make_func = functools.partial(make_tensor, dtype=dtype, device=device, low=-1, high=1)

    def nc_shape(rank: int, side: int):
        return tuple([N, C] + [side] * rank)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape with size fixed at compile time
        for rank in ranks:
            size_tuple = tuple([S_target] * rank)
            for align_corners in dyn_align_corners:
                yield OpDynamicInput(
                    op_compile_input=OpSampleInput(
                        op_input=ms.Tensor(shape=tuple([N, C] + [None] * rank), dtype=dtype),
                        op_args=(),
                        op_kwargs={
                            "size": size_tuple,
                            "scale_factor": None,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": None,
                        },
                        sample_name=f'{op_info.name}_dynamic_shape_compile_input_size_r{rank}',
                    ),
                    op_running_inputs=(
                        OpSampleInput(
                            op_input=make_func(shape=nc_shape(rank, D1)),
                            op_args=(),
                            op_kwargs={
                                "size": size_tuple,
                                "scale_factor": None,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": None,
                            },
                            sample_name=f'{op_info.name}_dynamic_shape_running_input_size_r{rank}',
                        ),
                        OpSampleInput(
                            op_input=make_func(shape=nc_shape(rank, D2)),
                            op_args=(),
                            op_kwargs={
                                "size": size_tuple,
                                "scale_factor": None,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": None,
                            },
                            sample_name=f'{op_info.name}_dynamic_shape_running_input_size_r{rank}',
                        ),
                    ),
                )

        # Dynamic shape with scale_factor fixed at compile time
        for rank in ranks:
            for align_corners in dyn_align_corners:
                for scale in (1.7, 0.6):
                    yield OpDynamicInput(
                        op_compile_input=OpSampleInput(
                            op_input=ms.Tensor(shape=tuple([N, C] + [None] * rank), dtype=dtype),
                            op_args=(),
                            op_kwargs={
                                "size": None,
                                "scale_factor": scale,
                                "mode": internal_mode,
                                "align_corners": align_corners,
                                "recompute_scale_factor": None,
                            },
                            sample_name=f'{op_info.name}_dynamic_shape_compile_input_scale_r{rank}',
                        ),
                        op_running_inputs=(
                            OpSampleInput(
                                op_input=make_func(shape=nc_shape(rank, D1)),
                                op_args=(),
                                op_kwargs={
                                    "size": None,
                                    "scale_factor": scale,
                                    "mode": internal_mode,
                                    "align_corners": align_corners,
                                    "recompute_scale_factor": None,
                                },
                                sample_name=f'{op_info.name}_dynamic_shape_running_input_scale_r{rank}',
                            ),
                            OpSampleInput(
                                op_input=make_func(shape=nc_shape(rank, D2)),
                                op_args=(),
                                op_kwargs={
                                    "size": None,
                                    "scale_factor": scale,
                                    "mode": internal_mode,
                                    "align_corners": align_corners,
                                    "recompute_scale_factor": None,
                                },
                                sample_name=f'{op_info.name}_dynamic_shape_running_input_scale_r{rank}',
                            ),
                        ),
                    )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank with size specified
        for rank in ranks:
            size_tuple = tuple([S_target] * rank)
            align_corners = dyn_align_corners[0]
            yield OpDynamicInput(
                op_compile_input=OpSampleInput(
                    op_input=ms.Tensor(shape=None, dtype=dtype),
                    op_args=(),
                    op_kwargs={
                        "size": size_tuple,
                        "scale_factor": None,
                        "mode": internal_mode,
                        "align_corners": align_corners,
                        "recompute_scale_factor": None,
                    },
                    sample_name=f'{op_info.name}_dynamic_rank_compile_input_r{rank}',
                ),
                op_running_inputs=(
                    OpSampleInput(
                        op_input=make_func(shape=nc_shape(rank, D1)),
                        op_args=(),
                        op_kwargs={
                            "size": size_tuple,
                            "scale_factor": None,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": None,
                        },
                        sample_name=f'{op_info.name}_dynamic_rank_running_input_r{rank}',
                    ),
                    OpSampleInput(
                        op_input=make_func(shape=nc_shape(rank, D2)),
                        op_args=(),
                        op_kwargs={
                            "size": size_tuple,
                            "scale_factor": None,
                            "mode": internal_mode,
                            "align_corners": align_corners,
                            "recompute_scale_factor": None,
                        },
                        sample_name=f'{op_info.name}_dynamic_rank_running_input_r{rank}',
                    ),
                ),
            )


def tensor_sum_ms(op_input, *op_args, **op_kwargs):
    return op_input.sum(*op_args, **op_kwargs)

def tensor_sum_torch(op_input, *op_args, **op_kwargs):
    return op_input.sum(*op_args, **op_kwargs)

def tensor_mean_ms(op_input, *op_args, **op_kwargs):
    return op_input.mean(*op_args, **op_kwargs)

def tensor_mean_torch(op_input, *op_args, **op_kwargs):
    return op_input.mean(*op_args, **op_kwargs)

def tensor_argmax_ms(op_input, *op_args, **op_kwargs):
    return op_input.argmax(*op_args, **op_kwargs)

def tensor_argmax_torch(op_input, *op_args, **op_kwargs):
    return op_input.argmax(*op_args, **op_kwargs)

def tensor_argmin_ms(op_input, *op_args, **op_kwargs):
    return op_input.argmin(*op_args, **op_kwargs)

def tensor_argmin_torch(op_input, *op_args, **op_kwargs):
    return op_input.argmin(*op_args, **op_kwargs)


def basic_sample_inputs_reduction_count_nonzero(op_info, dtype, device=None, **kwargs):
    """count_nonzero does not have keepdim parameter"""
    for sample_input in basic_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs):
        sample_input.op_kwargs.pop('keepdim', None)
        yield sample_input


def extra_sample_inputs_reduction_count_nonzero(op_info, dtype, device=None, **kwargs):
    """count_nonzero does not have keepdim parameter"""
    for sample_input in extra_reference_inputs_reduction_op_common_func(op_info, dtype, device, **kwargs):
        sample_input.op_kwargs.pop('keepdim', None)
        yield sample_input

def basic_sample_inputs_mint_pow(op_info: OpInfo, dtype=None, device=None, **kwargs):
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
        _input = make_func(input_shape, low=op_info.input_low, high=op_info.input_high, random_method='randn')
        _other = make_func(other_shape, low=op_info.other_low, high=op_info.other_high, random_method='randn')

        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_pow(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, device=device, dtype=dtype)

    shapes = (
        ((2, 2, 3, 2), (2, 2, 3, 2)),
        ((2, 2, 2, 3, 2), (2, 2, 2, 3, 2)),
        ((3, 2, 2, 2, 2, 2), (3, 2, 2, 2, 2, 2)),
        ((2, 2, 2, 2, 2, 2, 3), (2, 2, 2, 2, 2, 2, 3)),
        ((2, 2, 2, 2, 2, 4, 2, 2), (2, 2, 2, 2, 2, 4, 2, 2)),
    )
    for input_shape, other_shape in shapes:
        _input = make_func(input_shape, low=op_info.input_low, high=op_info.input_high, random_method='randn')
        _other = make_func(other_shape, low=op_info.other_low, high=op_info.other_high, random_method='randn')

        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            sample_name=op_info.name,
        )


def dynamic_sample_inputs_mint_pow(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 2D input, dim=1 static length, axis 0 static at compile time
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(5, None), dtype=dtype),  # non-dim axis (0) static
                op_args=(ms.Tensor(shape=(5, None), dtype=dtype),),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6), random_method='randn'),
                    op_args=(make_func(shape=(5, 6), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8), random_method='randn'),
                    op_args=(make_func(shape=(5, 8), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Dynamic shape case 2: 3D input, dim=0 (static), other axes dynamic only on input
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim axis static
                op_args=(ms.Tensor(shape=(6, None, 2), dtype=dtype),),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2), random_method='randn'),
                    op_args=(make_func(shape=(6, 3, 2), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2), random_method='randn'),
                    op_args=(make_func(shape=(6, 5, 2), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(ms.Tensor(shape=None, dtype=dtype),),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,), random_method='randn'),
                    op_args=(make_func(shape=(3,), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3), random_method='randn'),
                    op_args=(make_func(shape=(2, 3), random_method='randn'),),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )


def basic_sample_inputs_mint_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (((S, S, S), (1, 2)),
             ((S, S, S), (-1, 2)),
             ((S, S, S), (-1, -1)),
             ((S, S, S), (1, -1)),
             ((S, S), (-1, 2)),
             ((S,), (0, 2))
             )
    for shape, (dim, index) in cases:
        yield OpSampleInput(
            op_input=make_x(shape),
            op_args=(dim, index),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_x = functools.partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((2, 3, 4), 1, -2),
        ((2, 2, 3, 2), -1, 1),
        ((2, 2, 2, 3, 2), 3, 1),
        ((3, 2, 2, 2, 2, 2), 0, 1),
        ((2, 2, 2, 2, 2, 2, 3), -1, 1),
        ((2, 2, 2, 2, 2, 4, 2, 2), 5, 3),
    )
    for shape, dim, index in cases:
        yield OpSampleInput(
            op_input=make_x(shape),
            op_args=(dim, index),
            op_kwargs={},
            sample_name=op_info.name,
        )


def dynamic_sample_inputs_mint_select(op_info: OpInfo, dtype=None, device=None, **kwargs):
    make_func = functools.partial(make_tensor, dtype=dtype, device=device)

    if not kwargs.get("only_dynamic_rank", False):
        # Dynamic shape case 1: 2D input, dim=1 static length, axis 0 static at compile time
        dim = 1
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(5, None), dtype=dtype),  # non-dim axis (0) static
                op_args=(dim, mutable(0)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(dim, mutable(1)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8)),
                    op_args=(dim, mutable(2)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
            ),
        )

        # Dynamic shape case 2: 3D input, dim=0 (static), other axes dynamic only on input
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=(6, None, 2), dtype=dtype),  # dim axis static
                op_args=(dim, mutable(1)),  # index dim-axis length dynamic
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(dim, mutable(2)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(dim, mutable(3)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
            ),
        )

    if not kwargs.get("only_dynamic_shape", False):
        # Dynamic rank case: input/index with unknown rank at compile time.
        dim = 0
        yield OpDynamicInput(
            op_compile_input=OpSampleInput(
                op_input=ms.Tensor(shape=None, dtype=dtype),
                op_args=(dim, mutable(2)),
                op_kwargs={},
                sample_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(dim, mutable(0)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(dim, mutable(1)),
                    op_kwargs={},
                    sample_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
        )


def basic_sample_inputs_mint_one_hot(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    M = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else MEDIUM_DIM_SIZE
    L = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else LARGE_DIM_SIZE

    def make_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=dtype, low=low, high=high)

    shapes = ((), (S,), (L, M, S))
    num_classess = (-1, 10)

    for shape, num_classes in itertools.product(shapes, num_classess):
        yield OpSampleInput(
            op_input=make_input(
                shape,
                low=0,
                high=10 if num_classes == -1 else num_classes // 2,
            ),
            op_args=(),
            op_kwargs={"num_classes": num_classes},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_one_hot(op_info: OpInfo, dtype=None, device=None, **kwargs):
    def make_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=dtype, low=low, high=high)

    shapes = (
        ((2, 2, 3, 2), -1),
        ((2, 2, 2, 3, 2), 10),
        ((3, 2, 2, 2, 2, 2), -1),
        ((2, 2, 2, 2, 2, 2, 3), 10),
    )
    for shape, num_classes in shapes:
        yield OpSampleInput(
            op_input=make_input(
                shape,
                low=0,
                high=10 if num_classes == -1 else num_classes // 2,
            ),
            op_args=(),
            op_kwargs={"num_classes": num_classes},
            sample_name=op_info.name,
        )


def basic_sample_inputs_mint_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    shapes = ((S, S, S), (S, S), (S,), (),)
    make_tensor_partial = functools.partial(make_tensor, dtype=dtype, device=device)
    for shape in shapes:
        yield OpSampleInput(make_tensor_partial(shape))
        if len(shape) > 1:
            yield OpSampleInput(
                op_input=make_tensor_partial(shape),
                op_args=(),
                op_kwargs={"start_dim": 1, "end_dim": -1},
                sample_name=op_info.name,
            )


def extra_sample_inputs_mint_flatten(op_info: OpInfo, dtype=None, device=None, **kwargs):
    # shape x start_dim x end_dim
    cases = (
        ((5, 4, 0, 1, 3, 7), 1, 3),
        ((5, 4, 0, 1, 3, 7), 4, 5),
        ((5, 4, 1, 1, 3, 7), 2, 3),
        ((), 0, -1),
        ((1,), 0, -1),
        ((3, 7, 5), 1, 2),
        ((4, 5), 1, 1),
        ((1, 5, 5, 1, 5, 1, 5, 1), 0, 2),
        ((1, 5, 5, 1, 5, 1, 5, 1), 3, -1),
        ((1, 5, 5, 1, 5, 7, 5, 1), -2, -1),
        ((2, 4, 2), 0, 1),
        ((4, 2, 2), 1, 2),
        ((0, 3, 4, 5), 1, 3),
    )

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    for shape, start, end in cases:
        yield OpSampleInput(
            make_arg(shape),
            op_args=(start, end,),
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            make_arg(shape, discontiguous=True).transpose(0, -1),
            op_args=(start, end,),
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            make_arg(shape).transpose(0, -1),
            op_args=(start, end,),
            sample_name=op_info.name
        )


def basic_sample_inputs_mint_reshape(op_info: OpInfo, dtype=None, device=None, **kwargs):
    S = SMALL_DIM_SIZE if kwargs.get("only_small_tensor_size", False) else EXTRA_SMALL_DIM_SIZE
    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)

    cases = (
        ((S, S, S), (S * S, S)),
        ((S * S, S), (S, S, S)),
        ((S * S, S), (S, -1, S)),  # neg index
        ((S * S * 2, S), (S, -1)),  # neg index
        ((S,), (S,)),
        ((), ()),  # empty
        ((), (1,)),
    )

    for a, b in cases:
        yield OpSampleInput(
            op_input=make_arg(a),
            op_args=(b,),
            op_kwargs={},
            sample_name=op_info.name,
        )


def extra_sample_inputs_mint_reshape(op_info: OpInfo, dtype=None, device=None, **kwargs):
    cases = (
        ((125,), (25, 5)),
        ((25, 25), (1, 5, 5, 1, 5, 1, 5, 1)),
        ((16, 32), (2, 4, 1, 4, 4, 1, 4)),
        ((16, 12), (12, 16)),
        ((1, 16, 12), (12, 16)),
        ((1, 5, 1, 5), (25, 1)),
        ((2, 4, 2), (4, 4)),
        ((1, 4), (1, 1, 2, 1, 2)),
        ((3, 5, 7), (7, 5, 3)),
        ((1,), ()),  # empty
        ((5, 0, 2, 3), (5, 0, 2, 3)),
        ((2, 1, 0, 3, 1), (5, 0)),
        ((1,), ()),  # empty
        ((4, 5, 6), (4, 5, 6, 1, 1, 1)),
        ((), (1, 1, 1, 1)),  # empty
    )

    irreversible_cases = (
        ((), (-1,)),  # neg index, empty
        ((4, 7, 9, 1, 1), (1, 4, 3, -1, 1)),  # neg index
    )

    make_arg = functools.partial(make_tensor, dtype=dtype, device=device)
    for a, b in cases:
        yield OpSampleInput(
            op_input=make_arg(a),
            op_args=(b,),
            op_kwargs={},
            sample_name=op_info.name,
        )
        yield OpSampleInput(
            op_input=make_arg(b),
            op_args=(a,),
            op_kwargs={},
            sample_name=op_info.name,
        )

    for a, b in irreversible_cases:
        yield OpSampleInput(
            op_input=make_arg(a),
            op_args=(b,),
            op_kwargs={},
            sample_name=op_info.name,
        )


# op database
op_db: Dict[str, OpInfo] = {
    'mint.add': BinaryOpInfo(
        name='mint.add',
        op=mint.add,
        op_func_without_kwargs=add_ext_func_grad_without_kwargs,
        ref=torch.add,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.add(op_args[0], alpha=op_kwargs.get('alpha', 1)),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        op_basic_reference_inputs_func=basic_sample_inputs_add_sub_ext,
        op_dynamic_inputs_func=dynamic_sample_inputs_add_sub_ext,
        op_error_inputs_func=error_inputs_add_sub_ext_func,
    ),
    'mint.sub': BinaryOpInfo(
        name='mint.sub',
        op=mint.sub,
        op_func_without_kwargs=sub_ext_func_grad_without_kwargs,
        ref=torch.sub,
        # tensor_variant is now a unused parameter, may be removed in the future
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.sub(op_args[0], alpha=op_kwargs.get('alpha', 1)),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d != ms.bool_),
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        # On Ascend 910B, the bf16 results match PTA bitwise, but specified value has large deviation
        # when using random inputs, so we temporarily override the default loss value for bf16.
        default_loss_override={ms.bfloat16: 4e-2},
        op_basic_reference_inputs_func=basic_sample_inputs_add_sub_ext,
        op_dynamic_inputs_func=dynamic_sample_inputs_add_sub_ext,
        op_error_inputs_func=error_inputs_add_sub_ext_func,
    ),
    'mint.equal': BinaryOpInfo(
        name='mint.equal',
        op=mint.equal,
        op_func_without_kwargs=equal_func_grad,
        ref=torch.equal,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.equal(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.eq': BinaryOpInfo(
        name='mint.eq',
        op=mint.eq,
        op_func_without_kwargs=eq_func_grad,
        ref=torch.eq,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.eq(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.greater_equal': BinaryOpInfo(
        name='mint.greater_equal',
        op=mint.greater_equal,
        op_func_without_kwargs=greater_equal_func_grad,
        ref=torch.greater_equal,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.greater_equal(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.greater': BinaryOpInfo(
        name='mint.greater',
        op=mint.greater,
        op_func_without_kwargs=greater_func_grad,
        ref=torch.greater,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.greater(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.less_equal': BinaryOpInfo(
        name='mint.less_equal',
        op=mint.less_equal,
        op_func_without_kwargs=less_equal_func_grad,
        ref=torch.less_equal,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.less_equal(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.less': BinaryOpInfo(
        name='mint.less',
        op=mint.less,
        op_func_without_kwargs=less_func_grad,
        ref=torch.less,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.less(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.ne': BinaryOpInfo(
        name='mint.ne',
        op=mint.ne,
        op_func_without_kwargs=ne_func_grad,
        ref=torch.ne,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.ne(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_reference_inputs_binary_op_common_func,
        op_extra_reference_inputs_func=extra_reference_inputs_binary_op_common_func,
        op_dynamic_inputs_func=dynamic_inputs_binary_op_common_func,
        op_error_inputs_func=None,
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.maximum': BinaryOpInfo(
        name='mint.maximum',
        op=mint.maximum,
        op_func_without_kwargs=maximum_func_grad,
        ref=torch.maximum,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.equal(op_args[0]),
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.minimum': BinaryOpInfo(
        name='mint.minimum',
        op=mint.minimum,
        op_func_without_kwargs=minimum_func_grad,
        ref=torch.minimum,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.minimum(op_args[0]),
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'mint.div': BinaryOpInfo(
        name='mint.div',
        op=mint.div,
        op_func_without_kwargs=div_func_grad,
        ref=torch.div,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.div(op_args[0]),
        # Skip FP16 fwd/bwd on Ascend 910A/910B due to out-of-tolerance numerics vs PyTorch-CPU.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float16),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=True,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.mul': BinaryOpInfo(
        name='mint.mul',
        op=mint.mul,
        op_func_without_kwargs=mul_func_grad,
        ref=torch.mul,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.mul(op_args[0]),
        # Skip FP16 fwd/bwd on Ascend 910A/910B due to out-of-tolerance numerics vs PyTorch-CPU.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float16),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=True,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
        disable_large_value_tensor_inputs=True,
    ),
    'mint.repeat_interleave': OpInfo(
        name='mint.repeat_interleave',
        op=mint.repeat_interleave,
        op_func_without_kwargs=repeat_interleave_func_grad,
        ref=torch.repeat_interleave,
        dtypes_ascend=tuple(),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_repeat_interleave,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
    ),
    'mint.arange': OpInfo(
        name='mint.arange',
        op=mint.arange,
        ref=torch.arange,
        dtypes_ascend=tuple([ms.float32, ms.int32]),
        dtypes_ascend910b=tuple([ms.float32, ms.int32]),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_arange,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        is_differentiable=False,
    ),
    'Tensor.repeat_interleave': OpInfo(
        name='Tensor.repeat_interleave',
        op=tensor_repeat_interleave_ms,
        op_func_without_kwargs=tensor_repeat_interleave_func_grad,
        ref=tensor_repeat_interleave_torch,
        dtypes_ascend=tuple(),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_repeat_interleave,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
    ),
    'Tensor.repeat': OpInfo(
        name='Tensor.repeat',
        op=tensor_repeat_ms,
        ref=tensor_repeat_torch,
        # Repeat is shape-only; allow wide dtype sets. Exclude bf16 on generic Ascend for compatibility.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_basic_reference_inputs_func=basic_sample_inputs_tensor_repeat,
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
    ),
    'Tensor.maximum': BinaryOpInfo(
        name='Tensor.maximum',
        op=tensor_maximum_ms,
        ref=tensor_maximum_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.maximum(op_args[0]),
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'Tensor.minimum': BinaryOpInfo(
        name='Tensor.minimum',
        op=tensor_minimum_ms,
        ref=tensor_minimum_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.minimum(op_args[0]),
        #On Ascend 910A and 910B, float64 is not supported due to backward compatibility, so we need to exclude it.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float64),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float64),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
    ),
    'Tensor.mul': BinaryOpInfo(
        name='Tensor.mul',
        op=tensor_mul_ms,
        ref=tensor_mul_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.mul(op_args[0]),
        # Skip FP16 fwd/bwd on Ascend 910A/910B due to out-of-tolerance numerics vs PyTorch-CPU.
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16 and d != ms.float16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.float16),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        op_dynamic_inputs_func=None,
        op_error_inputs_func=None,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
        disable_large_value_tensor_inputs=True,
    ),
    'mint.floor_divide': BinaryOpInfo(
        name='mint.floor_divide',
        op=mint.floor_divide,
        op_func_without_kwargs=floor_divide_ext_func_grad_without_kwargs,
        ref=torch.floor_divide,
        is_differentiable=False,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.add(op_args[0]),
        # torch donen't support bool
        # ms.float16: Precision assertion failed.
        # ms.uint8: Error message EH9999.
        dtypes_ascend=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64,)),
        dtypes_ascend910b=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64,)),
        domain=((None, None), (1, None)),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.pow': BinaryOpInfo(
        name='mint.pow',
        op=mint.pow,
        op_func_without_kwargs=pow_ext_func_grad_without_kwargs,
        ref=torch.pow,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.add(op_args[0]),
        dtypes_ascend=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64,)),
        dtypes_ascend910b=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64, ms.bfloat16)),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_pow,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_pow,
        op_dynamic_inputs_func=None,
        domain=((1e-4, None), (1e-4, None)),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.tanh': UnaryOpInfo(
        name='mint.tanh',
        op=mint.tanh,
        ref=torch.tanh,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        #dtypes_cpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        #dtypes_gpu=tuple(d for d in dtypes_as_torch if (d.is_floating_point or d.is_complex) and d != ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        default_loss_override={ms.float16: 1e-3, ms.float32: 1e-4},
        # tanh has precision problem when converting input from half(fp16) to float
        convert_half_to_float=False,
    ),
    'Tensor.tanh': UnaryOpInfo(
        name='Tensor.tanh',
        op=tensor_tanh_ms,
        ref=tensor_tanh_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # tanh has precision problem when converting input from half(fp16) to float
        convert_half_to_float=False,
    ),
    'mint.nn.Tanh': UnaryOpInfo(
        name='mint.nn.Tanh',
        op=nn_tanh_ms,
        ref=nn_tanh_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # tanh has precision problem when converting input from half(fp16) to float
        convert_half_to_float=False,
    ),
    'mint.chunk': OpInfo(
        name='mint.chunk',
        op=mint.chunk,
        ref=torch.chunk,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_chunk,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_chunk,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_chunk, # mint.chunk limitedly supports dynamic cases
    ),
    'mint.gather': OpInfo(
        name='mint.gather',
        op=mint.gather,
        ref=torch.gather,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_gather,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_gather,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_gather,
    ),
    'mint.nn.functional.interpolate(mode="bilinear")': OpInfo(
        name='mint.nn.functional.interpolate(mode="bilinear")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        # On Ascend 910A, the results match PTA bitwise, but the deviation vs torch_cpu is large,
        # thus it should be considered unavailable.
        dtypes_ascend=(),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="bilinear"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="bilinear"),
    ),
    'mint.nn.functional.interpolate(mode="trilinear")': OpInfo(
        name='mint.nn.functional.interpolate(mode="trilinear")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.float16, ms.float32, ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="trilinear"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="trilinear"),
    ),
    'mint.nn.functional.interpolate(mode="bicubic")': OpInfo(
        name='mint.nn.functional.interpolate(mode="bicubic")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.float16, ms.float32),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="bicubic"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="bicubic"),
    ),
    'mint.nn.functional.interpolate(mode="linear")': OpInfo(
        name='mint.nn.functional.interpolate(mode="linear")',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        # On Ascend 910A, the results match PTA bitwise, but the deviation vs torch_cpu is large,
        # thus it should be considered unavailable.
        dtypes_ascend=(),
        dtypes_ascend910b=(ms.float16, ms.float32),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="linear"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="linear"),
    ),
    'mint.nn.functional.interpolate(mode="nearest")-1d': OpInfo(
        name='mint.nn.functional.interpolate(mode="nearest")-1d',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.uint8, ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.uint8, ms.float16, ms.float32, ms.float64, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="nearest1d"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="nearest1d"),
    ),
    'mint.nn.functional.interpolate(mode="nearest")-2d': OpInfo(
        name='mint.nn.functional.interpolate(mode="nearest")-2d',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.uint8, ms.float16, ms.float32),
        dtypes_ascend910b=(ms.uint8, ms.float16, ms.float32, ms.bfloat16),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="nearest2d"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="nearest2d"),
    ),
    'mint.nn.functional.interpolate(mode="nearest")-3d': OpInfo(
        name='mint.nn.functional.interpolate(mode="nearest")-3d',
        op=mint.nn.functional.interpolate,
        ref=torch.nn.functional.interpolate,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.float16, ms.float32, ms.float64),
        dtypes_cpu=(),
        dtypes_gpu=(),
        op_basic_reference_inputs_func=functools.partial(basic_sample_inputs_mint_interpolate, mode="nearest3d"),
        op_extra_reference_inputs_func=None,
        op_dynamic_inputs_func=functools.partial(dynamic_sample_inputs_mint_interpolate, mode="nearest3d"),
    ),
    'mint.mean': ReductionOpInfo(
        name='mint.mean',
        op=mint.mean,
        ref=torch.mean,
        # Todo: MindSpore additionally supports int dtype, while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if (d.is_floating_point or d.is_complex) and d != ms.bfloat16
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d.is_floating_point or d.is_complex
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: empty tensor case result not correct.
        op_extra_reference_inputs_func=skip_sample_inputs(
            extra_reference_inputs_reduction_op_common_func, 'empty'
        ),
        op_dynamic_inputs_func=None,
    ),
    'mint.argmax': ReductionOpInfo(
        name='mint.argmax',
        op=mint.argmax,
        ref=torch.argmax,
        # Todo: MindSpore additionally supports uint16/32/64 bool and complex
        # dtype, while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases, results are not correct.
        op_basic_reference_inputs_func=skip_sample_inputs(
            basic_reference_inputs_reduction_op_common_func, '_default_keepdim'
        ),
        op_extra_reference_inputs_func=skip_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', 'discontiguous_default', '_extremal_nan']
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'mint.argmin': ReductionOpInfo(
        name='mint.argmin',
        op=mint.argmin,
        ref=torch.argmin,
        # To do: MindSpore additionally supports uint16/32/64 dtype,
        # while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases,
        # results are not correct.
        op_basic_reference_inputs_func=skip_sample_inputs(
            basic_reference_inputs_reduction_op_common_func,
            '_default_keepdim'
        ),
        op_extra_reference_inputs_func=skip_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', 'discontiguous_default', '_extremal_nan']
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'mint.count_nonzero': ReductionOpInfo(
        name='mint.count_nonzero',
        op=mint.count_nonzero,
        ref=torch.count_nonzero,
        is_differentiable=False,
        # To do: MindSpore additionally supports uint16/32/64 dtype, while PyTorch does not.
        op_basic_reference_inputs_func=basic_sample_inputs_reduction_count_nonzero,
        op_extra_reference_inputs_func=extra_sample_inputs_reduction_count_nonzero,
        op_dynamic_inputs_func=None,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.argmax': ReductionOpInfo(
        name='Tensor.argmax',
        op=tensor_argmax_ms,
        ref=tensor_argmax_torch,
        # To do: MindSpore additionally supports uint16/32/64 bool and complex
        # dtype, while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(ms.float32,),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases,
        # results are not correct.
        op_basic_reference_inputs_func=skip_sample_inputs(
            basic_reference_inputs_reduction_op_common_func,
            ['_default', 'dim0_keepdim', 'dim_last']
        ),
        op_extra_reference_inputs_func=skip_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', '_default', '_extremal_nan']
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'Tensor.argmin': ReductionOpInfo(
        name='Tensor.argmin',
        op=tensor_argmin_ms,
        ref=tensor_argmin_torch,
        # To do: MindSpore additionally supports uint16/32/64 dtype,
        # while PyTorch does not.
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bfloat16, ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_ascend910b=tuple(
            d for d in dtypes_as_torch
            if d not in (ms.bool_, ms.complex64, ms.complex128)
        ),
        dtypes_cpu=(),
        dtypes_gpu=(),
        # Todo: When dim is default or empty tensor cases,
        # results are not correct.
        op_basic_reference_inputs_func=skip_sample_inputs(
            basic_reference_inputs_reduction_op_common_func,
            '_default'
        ),
        op_extra_reference_inputs_func=skip_sample_inputs(
            extra_reference_inputs_reduction_op_common_func,
            ['empty', '_default', '_extremal_nan']
        ),
        op_dynamic_inputs_func=None,
        supports_multiple_dims=False,
        is_differentiable=False,
    ),
    'Tensor.sum': ReductionOpInfo(
        name='Tensor.sum',
        op=tensor_sum_ms,
        ref=tensor_sum_torch,
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
        convert_half_to_float=True,
    ),
    'Tensor.mean': ReductionOpInfo(
        name='Tensor.mean',
        op=tensor_mean_ms,
        ref=tensor_mean_torch,
        op_dynamic_inputs_func=None,
        dtypes_ascend=(ms.float32,),
        dtypes_ascend910b=(ms.float32,),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.eq': BinaryOpInfo(
        name='Tensor.eq',
        op=tensor_eq_ms,
        ref=tensor_eq_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.eq(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.greater_equal': BinaryOpInfo(
        name='Tensor.greater_equal',
        op=tensor_greater_equal_ms,
        ref=tensor_greater_equal_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.greater_equal(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.greater': BinaryOpInfo(
        name='Tensor.greater',
        op=tensor_greater_ms,
        ref=tensor_greater_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.greater(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.less_equal': BinaryOpInfo(
        name='Tensor.less_equal',
        op=tensor_less_equal_ms,
        ref=tensor_less_equal_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.less_equal(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.less': BinaryOpInfo(
        name='Tensor.less',
        op=tensor_less_ms,
        ref=tensor_less_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.less(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.ne': BinaryOpInfo(
        name='Tensor.ne',
        op=tensor_ne_ms,
        ref=tensor_ne_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.ne(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.gt': BinaryOpInfo(
        name='Tensor.gt',
        op=tensor_gt_ms,
        ref=tensor_gt_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.gt(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.le': BinaryOpInfo(
        name='Tensor.le',
        op=tensor_le_ms,
        ref=tensor_le_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.le(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'Tensor.lt': BinaryOpInfo(
        name='Tensor.lt',
        op=tensor_lt_ms,
        ref=tensor_lt_torch,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.lt(op_args[0]),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=tuple(),
        dtypes_gpu=tuple(),
        is_differentiable=False,
        supports_left_python_scalar=False,
        supports_right_python_scalar=True,
        supports_both_python_scalar=False,
    ),
    'mint.floor': UnaryOpInfo(
        name='mint.floor',
        op=mint.floor,
        ref=torch.floor,
        dtypes_ascend=tuple(
            d for d in dtypes_as_torch if not d.is_complex and d not in [ms.bool_, ms.bfloat16]
        ),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bool_),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.ceil': UnaryOpInfo(
        name='mint.ceil',
        op=mint.ceil,
        ref=torch.ceil,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.float64),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.exp': UnaryOpInfo(
        name='mint.exp',
        op=mint.exp,
        ref=torch.exp,
        dtypes_ascend=(
            ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_
        ),
        dtypes_ascend910b=(
            ms.bfloat16, ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_
        ),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.log': UnaryOpInfo(
        name='mint.log',
        op=mint.log,
        ref=torch.log,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(1e-5, None),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.neg': UnaryOpInfo(
        name='mint.neg',
        op=mint.neg,
        ref=torch.neg,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_])),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.sigmoid': UnaryOpInfo(
        name='mint.sigmoid',
        op=mint.sigmoid,
        ref=torch.sigmoid,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.sqrt': UnaryOpInfo(
        name='mint.sqrt',
        op=mint.sqrt,
        ref=torch.sqrt,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(0, None),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.abs': UnaryOpInfo(
        name='mint.abs',
        op=mint.abs,
        ref=torch.abs,
        # torch donen't support bool
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d not in [ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bool_)),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.floor': UnaryOpInfo(
        name='Tensor.floor',
        op=tensor_floor_ms,
        ref=tensor_floor_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if not d.is_complex and d not in [ms.bool_, ms.bfloat16]),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex and d != ms.bool_),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.ceil': UnaryOpInfo(
        name='Tensor.ceil',
        op=tensor_ceil_ms,
        ref=tensor_ceil_torch,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64),
        dtypes_ascend910b=(ms.bfloat16, ms.float16, ms.float32, ms.float64),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.exp': UnaryOpInfo(
        name='Tensor.exp',
        op=tensor_exp_ms,
        ref=tensor_exp_torch,
        dtypes_ascend=(ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_),
        dtypes_ascend910b=(
            ms.bfloat16, ms.float16, ms.float32, ms.float64, ms.complex64, ms.complex128, ms.int64, ms.bool_
        ),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.log': UnaryOpInfo(
        name='Tensor.log',
        op=tensor_log_ms,
        ref=tensor_log_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(1e-5, None),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.neg': UnaryOpInfo(
        name='Tensor.neg',
        op=tensor_neg_ms,
        ref=tensor_neg_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (d not in [ms.int16, ms.uint8, ms.bool_])),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.sigmoid': UnaryOpInfo(
        name='Tensor.sigmoid',
        op=tensor_sigmoid_ms,
        ref=tensor_sigmoid_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.sqrt': UnaryOpInfo(
        name='Tensor.sqrt',
        op=tensor_sqrt_ms,
        ref=tensor_sqrt_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        domain=(0, None),
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.abs': UnaryOpInfo(
        name='Tensor.abs',
        op=tensor_abs_ms,
        ref=tensor_abs_torch,
        # torch donen't support bool_
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d not in [ms.bool_, ms.bfloat16])),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bool_)),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.square': UnaryOpInfo(
        name='Tensor.square',
        op=tensor_square_ms,
        ref=tensor_square_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.select': OpInfo(
        name='Tensor.select',
        op=tensor_select_ms,
        ref=tensor_select_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_select,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_select,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_select,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'Tensor.floor_divide': BinaryOpInfo(
        name='Tensor.floor_divide',
        op=tensor_floor_divide_ms,
        op_func_without_kwargs=tensor_floor_divide_ms,
        ref=tensor_floor_divide_torch,
        is_differentiable=False,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.add(op_args[0]),
        # torch donen't support bool
        # ms.float16: Precision assertion failed.
        # ms.uint8: Error message EH9999.
        dtypes_ascend=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64,)),
        dtypes_ascend910b=tuple((ms.int8, ms.int16, ms.int32, ms.int64, ms.float32, ms.float64,)),
        domain=((None, None), (1, None)),
        supports_left_python_scalar=False,
        supports_right_python_scalar=False,
        supports_both_python_scalar=False,
        disable_small_value_tensor_inputs=True,
        disable_large_value_tensor_inputs=True,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.ones_like': UnaryOpInfo(
        name='mint.ones_like',
        op=mint.ones_like,
        ref=torch.ones_like,
        is_differentiable=False,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if not d.is_complex),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.zeros_like': UnaryOpInfo(
        name='mint.zeros_like',
        op=mint.zeros_like,
        ref=torch.zeros_like,
        is_differentiable=False,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d not in [ms.uint32, ms.uint64, ms.bfloat16]),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d not in [ms.uint32, ms.uint64]),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.select': OpInfo(
        name='mint.select',
        op=mint.select,
        ref=torch.select,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_select,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_select,
        op_dynamic_inputs_func=dynamic_sample_inputs_mint_select,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.flatten': OpInfo(
        name='mint.flatten',
        op=mint.flatten,
        ref=torch.flatten,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_flatten,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_flatten,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.reshape': OpInfo(
        name='mint.reshape',
        op=mint.reshape,
        ref=torch.reshape,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_reshape,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_reshape,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.functional.one_hot': OpInfo(
        name='mint.nn.functional.one_hot',
        op=mint.nn.functional.one_hot,
        ref=torch.nn.functional.one_hot,
        dtypes_ascend=tuple((ms.int64,)),
        dtypes_ascend910b=tuple((ms.int64,)),
        op_basic_reference_inputs_func=basic_sample_inputs_mint_one_hot,
        op_extra_reference_inputs_func=extra_sample_inputs_mint_one_hot,
        op_dynamic_inputs_func=None,
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
}

all_op_db = list(op_db.keys())

binary_op_db = [
    'mint.add',
    'mint.sub',
    'mint.equal',
    'mint.eq',
    'mint.greater_equal',
    'mint.greater',
    'mint.less_equal',
    'mint.less',
    'mint.ne',
    'mint.maximum',
    'mint.minimum',
    'mint.div',
    'mint.mul',
    'Tensor.eq',
    'Tensor.greater_equal',
    'Tensor.greater',
    'Tensor.less_equal',
    'Tensor.less',
    'Tensor.ne',
    'Tensor.gt',
    'Tensor.le',
    'Tensor.lt',
    'Tensor.maximum',
    'Tensor.minimum',
    'Tensor.mul',
    'mint.floor_divide',
    'Tensor.floor_divide',
    'mint.pow',
]

unary_op_db = [
    'mint.tanh',
    'Tensor.tanh',
    'mint.nn.Tanh',
    'mint.floor',
    'mint.ceil',
    'mint.exp',
    'mint.log',
    'mint.neg',
    'mint.sigmoid',
    'mint.sqrt',
    'mint.abs',
    'Tensor.floor',
    'Tensor.ceil',
    'Tensor.exp',
    'Tensor.log',
    'Tensor.neg',
    'Tensor.sigmoid',
    'Tensor.sqrt',
    'Tensor.abs',
    'Tensor.square',
    'mint.ones_like',
    'mint.zeros_like',
]

other_op_db = [
    'mint.chunk',
    'mint.gather',
    'mint.nn.functional.interpolate(mode="bilinear")',
    'mint.nn.functional.interpolate(mode="trilinear")',
    'mint.nn.functional.interpolate(mode="bicubic")',
    'mint.nn.functional.interpolate(mode="linear")',
    'mint.nn.functional.interpolate(mode="nearest")-1d',
    'mint.nn.functional.interpolate(mode="nearest")-2d',
    'mint.nn.functional.interpolate(mode="nearest")-3d',
    'mint.repeat_interleave',
    'Tensor.repeat_interleave',
    'Tensor.repeat',
    'mint.arange',
    'mint.select',
    'Tensor.select',
    'mint.nn.functional.one_hot',
    'mint.flatten',
    'mint.reshape',
]

reduction_op_db = [
    'mint.mean',
    'mint.argmax',
    'mint.argmin',
    'mint.count_nonzero',
    'Tensor.mean',
    'Tensor.sum',
    'Tensor.argmax',
    'Tensor.argmin',
]

def get_op_info(op_name: str, *, op_database: Optional[Dict[str, OpInfo]] = None) -> OpInfo:
    """Return `OpInfo` by name from the provided or default database."""
    if op_name not in all_op_db:
        raise ValueError(f"op name {op_name} not found in op database")
    op_database = op_db if op_database is None else op_database
    return op_database[op_name]
