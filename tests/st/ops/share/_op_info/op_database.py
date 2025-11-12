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
import torch
import mindspore as ms
from mindspore import mint, mutable
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo, UnaryOpInfo
from tests.st.ops.share._op_info.op_info import basic_reference_inputs_binary_op_common_func
from tests.st.ops.share._op_info.op_common import dtypes_as_torch, dtypes_extra_uint
from tests.st.ops.share._op_info.op_common import SMALL_DIM_SIZE
from tests.st.ops.share._internal.utils import OpSampleInput, OpDynamicInput, OpErrorInput, make_tensor
from typing import Dict, Optional

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
            op_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': True},
            op_name=op_info.name
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
            op_name=op_info.name
        )
    else:
        yield OpSampleInput(
            op_input=_input,
            op_args=(_other,),
            op_kwargs={'alpha': False},
            op_name=op_info.name
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
                op_name=f'{op_info.name}_dynamic_shape_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5, 8, 5, 4)),
                    op_args=(make_func(shape=(5, 5, 8, 1, 4)),),
                    op_kwargs={"alpha": mutable(input_data=4.3, dynamic_len=False)},
                    op_name=f'{op_info.name}_dynamic_shape_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 8, 8, 4)),
                    op_args=(make_func(shape=(9, 9, 8, 1, 4)),),
                    op_kwargs={"alpha": mutable(input_data=-2.1, dynamic_len=False)},
                    op_name=f'{op_info.name}_dynamic_shape_running_input'
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
                op_name=f'{op_info.name}_dynamic_rank_compile_input'
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 5)),
                    op_args=(make_func(shape=(5, 5)),),
                    op_kwargs={"alpha": mutable(input_data=9.6, dynamic_len=False)},
                    op_name=f'{op_info.name}_dynamic_rank_running_input'
                ),
                OpSampleInput(
                    op_input=make_func(shape=(9, 9, 7)),
                    op_args=(make_func(shape=(9, 9, 7)),),
                    op_kwargs={"alpha": mutable(input_data=10.10, dynamic_len=False)},
                    op_name=f'{op_info.name}_dynamic_rank_running_input'
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
            op_name=op_info.name,
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
            op_name=op_info.name,
        ),
        op_error_type=TypeError,
        op_error_info='other is not tensor or number',
    )

# op_func_without_kwargs, used by gradient comparison if there are kwargs in op
def add_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.add(x, y, alpha=alpha)

def sub_ext_func_grad_without_kwargs(x, y, alpha=1):
    return mint.sub(x, y, alpha=alpha)

# wrap tensor method for tanh
def tensor_tanh_ms(op_input):
    return op_input.tanh()

def tensor_tanh_torch(op_input):
    return op_input.tanh()

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
            op_name=op_info.name,
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
            op_name=op_info.name,
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
                op_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3, 6)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_A",
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
                op_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(chunks, dim),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_B",
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
        op_name=op_info.name,
    )

    # 2D: dim=0 and dim=1
    x_shape = (S, S)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(S, S), high=x_shape[0])),
        op_kwargs={},
        op_name=op_info.name,
    )
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(1, make_index(shape=(S, S // 2), high=x_shape[1])),
        op_kwargs={},
        op_name=op_info.name,
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
        op_name=op_info.name,
    )

    # Empty index tensor case (1D input). Although 1D was in basic, this is a distinct edge case.
    x_shape = (S,)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(0,), high=1, dtype=ms.int32)),
        op_kwargs={},
        op_name=op_info.name,
    )

    # 3D: gather along middle dim (dim=1)
    x_shape = (2, 3, 4)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(1, make_index(shape=(2, 2, 4), high=x_shape[1])),
        op_kwargs={},
        op_name=op_info.name,
    )

    # 4D: negative dim (-1)
    x_shape = (2, 2, 3, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(-1, make_index(shape=(2, 2, 3, 1), high=x_shape[-1])),
        op_kwargs={},
        op_name=op_info.name,
    )

    # 5D: dim=3, non-dim axes of index <= input
    x_shape = (2, 2, 2, 3, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(3, make_index(shape=(2, 2, 2, 2, 2), high=x_shape[3])),
        op_kwargs={},
        op_name=op_info.name,
    )

    # 6D: dim=0
    x_shape = (3, 2, 2, 2, 2, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(0, make_index(shape=(2, 2, 2, 2, 2, 2), high=x_shape[0])),
        op_kwargs={},
        op_name=op_info.name,
    )

    # 7D: last dim
    x_shape = (2, 2, 2, 2, 2, 2, 3)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(-1, make_index(shape=(2, 2, 2, 2, 2, 2, 2), high=x_shape[-1])),
        op_kwargs={},
        op_name=op_info.name,
    )

    # 8D: dim=5
    x_shape = (2, 2, 2, 2, 2, 4, 2, 2)
    yield OpSampleInput(
        op_input=make_x(x_shape),
        op_args=(5, make_index(shape=(2, 2, 2, 2, 2, 2, 2, 2), high=x_shape[5])),
        op_kwargs={},
        op_name=op_info.name,
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
                op_name=f"{op_info.name}_dynamic_shape_compile_input_A",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(5, 6)),
                    op_args=(dim, make_tensor(shape=(5, 3), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_A",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(5, 8)),
                    op_args=(dim, make_tensor(shape=(5, 4), dtype=ms.int64, device=device, low=0, high=8)),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_A",
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
                op_name=f"{op_info.name}_dynamic_shape_compile_input_B",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(6, 3, 2)),
                    op_args=(dim, make_tensor(shape=(3, 2, 2), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_B",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(6, 5, 2)),
                    op_args=(dim, make_tensor(shape=(4, 2, 2), dtype=ms.int64, device=device, low=0, high=6)),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_shape_running_input_B",
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
                op_name=f"{op_info.name}_dynamic_rank_compile_input",
            ),
            op_running_inputs=(
                OpSampleInput(
                    op_input=make_func(shape=(3,)),
                    op_args=(dim, make_tensor(shape=(2,), dtype=ms.int64, device=device, low=0, high=3)),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
                OpSampleInput(
                    op_input=make_func(shape=(2, 3)),
                    op_args=(dim, make_tensor(shape=(2, 3), dtype=ms.int64, device=device, low=0, high=2)),
                    op_kwargs={},
                    op_name=f"{op_info.name}_dynamic_rank_running_input",
                ),
            ),
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
        op_basic_reference_inputs_func=basic_sample_inputs_add_sub_ext,
        op_dynamic_inputs_func=dynamic_sample_inputs_add_sub_ext,
        op_error_inputs_func=error_inputs_add_sub_ext_func,
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
    ),
    'Tensor.tanh': UnaryOpInfo(
        name='Tensor.tanh',
        op=tensor_tanh_ms,
        ref=tensor_tanh_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
    ),
    'mint.nn.Tanh': UnaryOpInfo(
        name='mint.nn.Tanh',
        op=nn_tanh_ms,
        ref=nn_tanh_torch,
        dtypes_ascend=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.bfloat16 and d != ms.float64)),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if (not d.is_complex and d != ms.float64)),
        dtypes_cpu=(),
        dtypes_gpu=(),
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
}

all_op_db = list(op_db.keys())

binary_op_db = [
    'mint.add',
    'mint.sub',
]

unary_op_db = [
    'mint.tanh',
    'Tensor.tanh',
    'mint.nn.Tanh',
]

other_op_db = [
    'mint.chunk',
    'mint.gather'
]

def get_op_info(op_name: str, *, op_database: Optional[Dict[str, OpInfo]] = None) -> OpInfo:
    """Return `OpInfo` by name from the provided or default database."""
    if op_name not in all_op_db:
        raise ValueError(f"op name {op_name} not found in op database")
    op_database = op_db if op_database is None else op_database
    return op_database[op_name]
