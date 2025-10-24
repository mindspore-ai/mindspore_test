# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import torch
import random
import numpy as np
import mindspore as ms
from mindspore import ops, mutable
from mindspore.common.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.ops.share._internal.reduction_ops import ReductionOpsFactory
from tests.st.ops.share._internal.utils import make_tensor, OpSampleInput


# The following test cases are migrated from MindSporeTest.operations.test_f_sum.py
@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_float_complex_forward_grad(mode):
    '''
    Feature: ops.sum
    Description: Float and complex dtypes, various shapes/axes/keepdim and dtype casts; compare forward and gradients.
    Expectation: Outputs and gradients match reference.
    '''
    def float_complex_sample_inputs_func():
        sample_inputs = []
        # complex64 4D, random dim(list) and keepdim=False, dtype=complex64
        dims = (-4, -3, -2, -1, 0, 1, 2, 3)
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((3, 7, 6, 8), ms.complex64),
            op_args=(np.random.choice(dims, size=1).tolist(), False),
            op_kwargs=dict(dtype=ms.complex64),
            op_name='sum_complex64_4d'
        ))
        # complex128 2D, dim=0, keepdim=False, dtype=complex128
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((3, 7), ms.complex128),
            op_args=(0, False),
            op_kwargs=dict(dtype=ms.complex128),
            op_name='sum_complex128_2d'
        ))
        # float32 7D, random dim/keepdim, cast to float64
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((3, 7, 6, 8, 2, 3, 4), ms.float32),
            op_args=(random.randint(-7, 6), random.choice([True, False])),
            op_kwargs=dict(dtype=ms.float64),
            op_name='sum_float32_7d_to_float64'
        ))
        # float32 1D, dim=-1, keepdim=False, dtype=float32
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((5,), ms.float32),
            op_args=(-1, False),
            op_kwargs=dict(dtype=ms.float32),
            op_name='sum_float32_1d'
        ))
        # float16 2D, dim=-1, keepdim=True, dtype=None
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((9, 5), ms.float16),
            op_args=(-1, True),
            op_kwargs=dict(dtype=None),
            op_name='sum_float16_2d_keepdim'
        ))
        # float64 7D, dim=5, keepdim=False, dtype=None
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((5, 7, 8, 7, 9, 8, 4), ms.float64),
            op_args=(5, False),
            op_kwargs=dict(dtype=None),
            op_name='sum_float64_7d'
        ))
        return sample_inputs

    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        sample_inputs_func=float_complex_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_integer_forward(mode):
    '''
    Feature: ops.sum
    Description: Integer dtypes forward-only across shapes/dims/keepdim and output dtype conversions.
    Expectation: Outputs match reference.
    '''
    def integer_forward_sample_inputs_func():
        sample_inputs = []
        # int32 6D, random dim/keepdim, keep dtype as int32
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((3, 7, 6, 8, 2, 3), ms.int32),
            op_args=(random.randint(-6, 5), random.choice([True, False])),
            op_kwargs=dict(dtype=ms.int32),
            op_name='sum_int32_6d_forward'
        ))
        # int32 3D, dim=0, keepdim=False, cast to int64
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((7, 3, 4), ms.int32, -500, 500, random_method='randint'),
            op_args=(0, False),
            op_kwargs=dict(dtype=ms.int64),
            op_name='sum_int32_3d_to_int64'
        ))
        # int64 4D, dim=1, keepdim=True, cast to float64
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((3, 9, 5, 4), ms.int64, -500, 500, random_method='randint'),
            op_args=(1, True),
            op_kwargs=dict(dtype=ms.float64),
            op_name='sum_int64_4d_to_float64'
        ))
        # int16 6D, dim=-3, keepdim=False, keep dtype int16
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((4, 8, 3, 9, 5, 7), ms.int16, -500, 500, random_method='randint'),
            op_args=(-3, False),
            op_kwargs=dict(dtype=ms.int16),
            op_name='sum_int16_6d_forward'
        ))
        # int8 7D, dim=3, keepdim=True, cast to int16
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((9, 5, 7, 5, 3, 3, 8), ms.int8, -5, 5, random_method='randint'),
            op_args=(3, True),
            op_kwargs=dict(dtype=ms.int16),
            op_name='sum_int8_7d_to_int16'
        ))
        return sample_inputs

    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        sample_inputs_func=integer_forward_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()



@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_tuple_dim_6d(mode):
    '''
    Feature: ops.sum
    Description: float64 6D tensor; reduce over tuple dims (1,2,3,4); forward and gradient.
    Expectation: Outputs and gradients match reference.
    '''
    x = make_tensor((3, 7, 6, 8, 2, 3), ms.float64)
    dim = (1, 2, 3, 4)
    keepdim = random.choice([True, False])
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=ms.float64),
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_input_x_extra_uint_dtypes(mode):
    """
    Feature: sum op with extra uint dtypes.
    Description: forward and gradient comparison on uint16, uint32, uint64.
    Ported from MindSporeTest: test_f_sum_input_x_5d_uint16_dim_len3_true_mstype_int8
                               test_f_sum_input_x_2d_uint32_dim_len3_false_mstype_float16
                               test_f_sum_input_x_6d_uint64_dim_len3_false_mstype_uint64
    Expectation: outputs and gradients match reference.
    """
    def sum_extra_uint_dtypes_sample_inputs_func():
        sample_inputs = []
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((8, 4, 9, 3, 8), ms.uint16, 0, 5, random_method='randint'),
            op_args=(3, True),
            op_kwargs=dict(dtype=ms.int8),
            op_name='sum_uint16_sample_input',
        ))
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((7, 3), ms.uint32, 0, 100, random_method='randint'),
            op_args=(0, False),
            op_kwargs=dict(dtype=ms.float16),
            op_name='sum_uint32_sample_input',
        ))
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((5, 7, 8, 7, 9, 8), ms.uint64, 0, 1000, random_method='randint'),
            op_args=(-3, False),
            op_kwargs=dict(dtype=ms.int64),
            op_name='sum_uint64_sample_input',
        ))
        return sample_inputs

    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        sample_inputs_func=sum_extra_uint_dtypes_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()



@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_input_x_1d_uint8_0_true_mstype_int64(mode):
    '''
    Feature: ops.sum
    Description: uint8 1D tensor reduce at last dim, keepdim=True, dtype=int64; forward-only.
    Expectation: Outputs match reference.
    '''
    input_x = make_tensor((9,), ms.uint8, 0, 20, random_method='randint')
    dim = -1
    keepdim = True
    dtype = ms.int64
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()



@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_input_x_4d_bool__dim_len2_true_mstype_float64(mode):
    '''
    Feature: ops.sum
    Description: bool 4D tensor reduce at dim=3, keepdim=True, dtype=float64; forward-only.
    Expectation: Outputs match reference.
    '''
    input_x = make_tensor((8, 4, 9, 3), ms.bool)
    dim = 3
    keepdim = True
    dtype = ms.float64
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_input_x_1d_float32_dim_len2_false_none(mode):
    '''
    Feature: ops.sum
    Description: float32 1D tensor reduce at last dim, keepdim=False, dtype=None; forward and gradient.
    Expectation: Outputs and gradients match reference.
    '''
    input_x = make_tensor((7,), ms.float32)
    dim = -1
    keepdim = False
    dtype = None
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_input_x_not_tensor(mode):
    '''
    Feature: ops.sum
    Description: input_x is not a Tensor; validate type checking.
    Expectation: Raises TypeError.
    '''
    input_x = 1.0
    dim = -1
    keepdim = False
    dtype = ms.uint8
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_dim_float(mode):
    '''
    Feature: ops.sum
    Description: dim is float type; validate argument type checking.
    Expectation: Raises TypeError.
    '''
    input_x = make_tensor((5,), ms.float32)
    dim = 1.0
    keepdim = False
    dtype = ms.uint8
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_keep_dims_float(mode):
    '''
    Feature: ops.sum
    Description: keepdim is float type; validate argument type checking.
    Expectation: Raises TypeError.
    '''
    input_x = make_tensor((5,), ms.float32)
    dim = -1
    keepdim = 1.0
    dtype = ms.uint8
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_dtype_float(mode):
    '''
    Feature: ops.sum
    Description: dtype is float value; validate argument type checking.
    Expectation: Raises TypeError.
    '''
    input_x = make_tensor((5,), ms.float32)
    dim = -1
    keepdim = False
    dtype = 1.0
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_dim_out_range(mode):
    '''
    Feature: ops.sum
    Description: dim out of valid range; validate dimension checking.
    Expectation: Raises ValueError.
    '''
    input_x = make_tensor((5,), ms.float32)
    dim = -input_x.ndim - 2
    keepdim = False
    dtype = ms.uint8
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    with pytest.raises(ValueError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sum_input_5d_random(mode):
    '''
    Feature: ops.sum
    Description: bfloat16 5D tensor reduce at dim=-2, keepdim=True; forward and gradient.
    Expectation: Outputs and gradients match reference.
    '''
    input_x = make_tensor((9, 7, 4, 9, 5), ms.bfloat16)
    dim = -2
    keepdim = True
    dtype = ms.bfloat16
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        op_input=input_x,
        op_args=(dim, keepdim),
        op_kwargs=dict(dtype=dtype),
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sum_dynamic_shape(mode):
    '''
    Feature: ops.sum
    Description: Dynamic shape with fixed rank (1D). dim and keepdim are mutable; dtype=float32; compare
    forward and gradients across shapes.
    Expectation: Outputs and gradients match reference.
    '''
    def sum_dynamic_shape_sample_inputs_func():
        sample_inputs = []
        compile_input = OpSampleInput(
            op_input=ms.Tensor(shape=(None,), dtype=ms.float32),
            op_args=(mutable(input_data=0, dynamic_len=False), mutable(input_data=False, dynamic_len=False)),
            op_kwargs=dict(dtype=ms.float32),
            op_name='sum_compile_input'
        )
        sample_inputs.append(compile_input)
        shapes = [
            (3,),
            (5,),
        ]
        for shape in shapes:
            sample_inputs.append(OpSampleInput(
                op_input=make_tensor(shape, ms.float32),
                op_args=(mutable(input_data=0, dynamic_len=False), mutable(input_data=False, dynamic_len=False)),
                op_kwargs=dict(dtype=ms.float32),
                op_name='sum_running_input'
            ))
        return sample_inputs
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        sample_inputs_func=sum_dynamic_shape_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_dynamic_shape_cmp()
    fact.grad_dynamic_shape_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sum_dynamic_rank(mode):
    '''
    Feature: ops.sum
    Description: Dynamic rank (shape=None). dim and keepdim are mutable; shapes include 1D/3D/scalar; compare
    forward and gradients across ranks.
    Expectation: Outputs and gradients match reference.
    '''
    def sum_dynamic_shape_sample_inputs_func():
        sample_inputs = []
        compile_input = OpSampleInput(
            op_input=ms.Tensor(shape=None, dtype=ms.float32),
            op_args=(mutable(input_data=0, dynamic_len=False), mutable(input_data=False, dynamic_len=False)),
            op_kwargs=dict(dtype=ms.float32),
            op_name='sum_compile_input'
        )
        sample_inputs.append(compile_input)
        shapes = [
            (3,),
            (5, 2, 3),
            (),
        ]
        for shape in shapes:
            sample_inputs.append(OpSampleInput(
                op_input=make_tensor(shape, ms.float32),
                op_args=(mutable(input_data=0, dynamic_len=False), mutable(input_data=False, dynamic_len=False)),
                op_kwargs=dict(dtype=ms.float32),
                op_name='sum_running_input'
            ))
        return sample_inputs
    fact = ReductionOpsFactory(
        op=ops.sum,
        ref=torch.sum,
        sample_inputs_func=sum_dynamic_shape_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_dynamic_shape_cmp()
    fact.grad_dynamic_shape_cmp()
