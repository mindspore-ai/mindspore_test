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
from tests.st.ops.share._op_info.op_info import OpInfo, BinaryOpInfo
from tests.st.ops.share._op_info.op_info import sample_inputs_binary_op_func
from tests.st.ops.share._op_info.op_common import dtypes_as_torch, dtypes_extra_uint
from tests.st.ops.share._op_info.op_common import SMALL_DIM_SIZE
from tests.st.ops.share._internal.utils import OpSampleInput, OpDynamicInput, OpErrorInput, make_tensor
from typing import Dict, Optional

# op_sample_inputs_func for ops
def sample_inputs_add_sub_ext_func(
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
    yield from sample_inputs_binary_op_func(op_info, dtype, device, **kwargs)

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
def dynamic_inputs_add_sub_ext_func(
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

# op_func_grad, used by gradient comparison if there are kwargs in op
def add_ext_func_grad(x, y, alpha=1):
    return mint.add(x, y, alpha=alpha)

def sub_ext_func_grad(x, y, alpha=1):
    return mint.sub(x, y, alpha=alpha)

# op database
op_db: Dict[str, OpInfo] = {
    'add_ext': BinaryOpInfo(
        name='add_ext',
        op=mint.add,
        op_func_grad=add_ext_func_grad,
        ref=torch.add,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.add(op_args[0], alpha=op_kwargs.get('alpha', 1)),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16),
        dtypes_ascend910b=dtypes_as_torch,
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        op_sample_inputs_func=sample_inputs_add_sub_ext_func,
        op_dynamic_inputs_func=dynamic_inputs_add_sub_ext_func,
        op_error_inputs_func=error_inputs_add_sub_ext_func,
    ),
    'sub_ext': BinaryOpInfo(
        name='sub_ext',
        op=mint.sub,
        op_func_grad=sub_ext_func_grad,
        ref=torch.sub,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.sub(op_args[0], alpha=op_kwargs.get('alpha', 1)),
        dtypes_ascend=tuple(d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_),
        dtypes_ascend910b=tuple(d for d in dtypes_as_torch if d != ms.bool_),
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16 and d != ms.bool_] + list(dtypes_extra_uint)),
        op_sample_inputs_func=sample_inputs_add_sub_ext_func,
        op_dynamic_inputs_func=dynamic_inputs_add_sub_ext_func,
        op_error_inputs_func=error_inputs_add_sub_ext_func,
    ),
}

all_op_db = list(op_db.keys())

binary_op_db = [
    'add_ext',
    'sub_ext',
]

def get_op_info(op_name: str, *, op_database: Optional[Dict[str, OpInfo]] = None) -> OpInfo:
    """Return `OpInfo` by name from the provided or default database."""
    if op_name not in all_op_db:
        raise ValueError(f"op name {op_name} not found in op database")
    op_database = op_db if op_database is None else op_database
    return op_database[op_name]
