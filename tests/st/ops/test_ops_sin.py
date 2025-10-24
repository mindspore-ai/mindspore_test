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
"""Tests for sin operation and Mint/ops frontends.

Covers forward/backward, vmap, dynamic shape/rank, dtype and special values.
"""
import pytest
import torch
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.mint import sin
from mindspore.common.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.share._internal.elementwise_ops import ElementwiseOpsFactory
from tests.st.ops.share._internal.utils import make_tensor, make_tensor_with_np_array, OpSampleInput


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.sin(x)


def generate_expect_backward_output(x):
    return np.cos(x)


@test_utils.run_with_cell
def sin_forward_func(x):
    return sin(x)


@test_utils.run_with_cell
def sin_backward_func(x):
    return ms.grad(sin_forward_func, (0))(x)


@test_utils.run_with_cell
def sin_vmap_func(x):
    return ops.vmap(sin_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_sin_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function sin forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((64, 224), np.float32)
    output = sin_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((2, 3, 4, 5), np.float32)
    output2 = sin_backward_func(ms.Tensor(x2))
    expect2 = generate_expect_backward_output(x2)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_sin_forward_case01(context_mode):
    """
    Feature: pyboost function.
    Description: test function sin forward add cases.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((384, 128), np.float32)
    output = sin_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_sin_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function sin vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = sin_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_ops_sin_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function sin with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5, 6), np.float32)

    TEST_OP(sin_forward_func, [[ms.Tensor(x)], [ms.Tensor(y)]])


# The following test cases are migrated from MindSporeTest.operations.test_f_sin
def tensor_sin(x):
    return x.sin()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sin_input_nd_float32(mode):
    """
    Feature: elementwise sin op.
    Description: compare MindSpore and reference outputs for float32 with 0D and high-rank inputs.
    Ported from MindSporeTest: test_f_sin_input_0d_float32
                               test_f_sin_float32_8d_1x2x3x4x5x6x7x8_random
    Expectation: results are close within tolerance.
    """
    shapes = [
        (),
        (1, 2, 3, 4, 5, 6, 7, 8),
    ]
    fact = ElementwiseOpsFactory(
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)
    fact.test_elementwise_op_nd(shapes, ms.float32)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sin_float_complex_forward_grad(mode):
    """
    Feature: elementwise sin op.
    Description: float16/float64/complex64 inputs; forward and gradient comparison.
    Expectation: outputs and gradients match reference.
    """
    def float_complex_sample_inputs_func():
        sample_inputs = []
        # float16 1D
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((8,), ms.float16),
            op_name='sin_float16_1d',
        ))
        # float64 6D
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((3, 5, 8, 5, 7, 3), ms.float64),
            op_name='sin_float64_6d',
        ))
        # complex64 7D
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((8, 1, 5, 3, 4, 1, 5), ms.complex64, low=-10, high=10, random_method='randint'),
            op_name='sin_complex64_7d',
        ))
        return sample_inputs

    fact = ElementwiseOpsFactory(
        op=ops.sin,
        ref=torch.sin,
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
def test_tensor_sin_float_forward_grad(mode):
    """
    Feature: tensor method sin.
    Description: float32/float64 inputs with multiple shapes; forward and gradient comparison.
    Expectation: outputs and gradients match reference.
    """
    def tensor_float_sample_inputs_func():
        sample_inputs = []
        # float32 3D
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((3, 2, 4), ms.float32),
            op_name='tensor_sin_float32_3d',
        ))
        # float64 ND shapes
        for shape in [
                (1, 2, 3, 4, 5, 6),
                (2, 3, 7, 4, 3, 9, 3, 4),
                (2, 0),
        ]:
            sample_inputs.append(OpSampleInput(
                op_input=make_tensor(shape, ms.float64),
                op_name='tensor_sin_float64_nd',
            ))
        return sample_inputs

    fact = ElementwiseOpsFactory(
        op=tensor_sin,
        ref=tensor_sin,
        sample_inputs_func=tensor_float_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sin_integer_bool_forward(mode):
    """
    Feature: elementwise sin op.
    Description: forward comparison on integer/boolean inputs across shapes and dtypes.
    Expectation: results match reference (no grad for non-float dtypes).
    """
    def integer_bool_sample_inputs_func():
        sample_inputs = []
        # uint8 0D scalar
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((), ms.uint8, low=0, high=255, random_method='randint'),
            op_name='sin_uint8_0d',
        ))
        # int32 2D
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((7, 7), ms.int32),
            op_name='sin_int32_2d',
        ))
        # int64 3D with randint
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((5, 3, 3), ms.int64, low=-50, high=50, random_method='randint'),
            op_name='sin_int64_3d',
        ))
        # int8 4D with randint bounds
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((4, 8, 9, 9), ms.int8, low=-128, high=127, random_method='randint'),
            op_name='sin_int8_4d',
        ))
        # bool 6D
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor((4, 4, 3, 4, 4, 9), ms.bool_),
            op_name='sin_bool_6d',
        ))
        return sample_inputs

    fact = ElementwiseOpsFactory(
        op=ops.sin,
        ref=torch.sin,
        sample_inputs_func=integer_bool_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sin_input_3d_int64(mode):
    """
    Feature: elementwise sin op.
    Description: forward comparison on 3D int64 input using randint generation.
    Expectation: results are close within tolerance.
    """
    fact = ElementwiseOpsFactory(
        op_input=make_tensor((5, 3, 3), ms.int64, low=-50, high=50, random_method='randint'),
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sin_input_4d_int8(mode):
    """
    Feature: elementwise sin op.
    Description: forward comparison on 4D int8 input with specified randint bounds.
    Expectation: results are close within tolerance.
    """
    fact = ElementwiseOpsFactory(
        op_input=make_tensor((4, 8, 9, 9), ms.int8, low=-128, high=127, random_method='randint'),
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sin_input_0d_uint8(mode):
    """
    Feature: elementwise sin op.
    Description: forward comparison on 0D uint8 scalar input.
    Expectation: results are close within tolerance.
    """
    fact = ElementwiseOpsFactory(
        op_input=make_tensor((), ms.uint8, low=0, high=255, random_method='randint'),
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sin_input_5d_complex128(mode):
    """
    Feature: elementwise sin op.
    Description: forward comparison on 5D complex128 input, real/imag from randint range.
    Expectation: results are close within tolerance.
    """
    fact = ElementwiseOpsFactory(
        op_input=make_tensor((1, 4, 7, 1, 3), ms.complex128, low=-100, high=100, random_method='randint'),
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sin_bool_6d_4x4x3x4x4x9_random_forward(mode):
    """
    Feature: elementwise sin op.
    Description: forward comparison on 6D boolean input.
    Expectation: results match reference (no grad for bool).
    """
    fact = ElementwiseOpsFactory(
        op_input=make_tensor((4, 4, 3, 4, 4, 9), ms.bool_),
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sin_input_not_tensor(mode):
    """
    Feature: elementwise sin op input validation.
    Description: verify non-tensor input raises expected exception across modes.
    Expectation: raises RuntimeError/ValueError/TypeError as validated.
    """
    fact = ElementwiseOpsFactory(
        op_input=1.0,
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)

    with pytest.raises((RuntimeError, ValueError, TypeError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_sin_input_dtype_uint32(mode):
    """
    Feature: elementwise sin op input dtype checking.
    Description: verify unsupported dtype uint32 raises expected exception.
    Expectation: raises RuntimeError/ValueError/TypeError as validated.
    """
    fact = ElementwiseOpsFactory(
        op_input=make_tensor((), ms.uint32, low=0, high=1000, random_method='randint'),
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)

    with pytest.raises((RuntimeError, ValueError, TypeError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sin_special_values_forward_grad(mode):
    """
    Feature: elementwise sin op special values.
    Description: forward/grad comparison with inputs containing NaN and Inf.
    Expectation: outputs and gradients match reference handling of NaN/Inf.
    """
    def special_values_sample_inputs_func():
        sample_inputs = []
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((7, 3, 9), np.nan), ms.float32),
            op_name='sin_nan_float32',
        ))
        sample_inputs.append(OpSampleInput(
            op_input=make_tensor_with_np_array(np.full((4,), np.inf), ms.float64),
            op_name='sin_inf_float64',
        ))
        return sample_inputs

    fact = ElementwiseOpsFactory(
        op=ops.sin,
        ref=torch.sin,
        sample_inputs_func=special_values_sample_inputs_func,
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
def test_dynamic_shape_f_sin_float32(mode):
    """
    Feature: elementwise sin op dynamic shape.
    Description: compile with dynamic shape and run multiple concrete shapes for forward/grad.
    Expectation: results and gradients match reference for all shapes.
    """
    def dynamic_shape_sample_inputs_func():
        sample_inputs = []
        compile_input = OpSampleInput(
            op_input=ms.Tensor(shape=(None, None, None, None, None, None, None), dtype=ms.float32),
            op_name='sin_compile_input'
        )
        sample_inputs.append(compile_input)
        shapes = [
            (3, 9, 9, 5, 7, 8, 4),
            (4, 7, 9, 3, 5, 9, 9),
            (2, 2, 3, 4, 6, 7, 3),
        ]
        for shape in shapes:
            sample_inputs.append(OpSampleInput(
                op_input=make_tensor(shape, ms.float32),
                op_name='sin_running_input'
            ))
        return sample_inputs

    fact = ElementwiseOpsFactory(
        op=ops.sin,
        ref=torch.sin,
        sample_inputs_func=dynamic_shape_sample_inputs_func,
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
def test_dynamic_rank_f_sin_float16(mode):
    """
    Feature: elementwise sin op dynamic rank.
    Description: compile with dynamic rank and run multiple concrete ranks for forward/grad.
    Expectation: results and gradients match reference for all ranks.
    """
    def dynamic_shape_sample_inputs_func():
        sample_inputs = []
        compile_input = OpSampleInput(
            op_input=ms.Tensor(shape=None, dtype=ms.float16),
            op_name='sin_compile_input'
        )
        sample_inputs.append(compile_input)
        shapes = [
            (3, 9, 9),
            (4, 7, 9, 3),
            (2,),
        ]
        for shape in shapes:
            sample_inputs.append(OpSampleInput(
                op_input=make_tensor(shape, ms.float16),
                op_name='sin_running_input'
            ))
        return sample_inputs

    fact = ElementwiseOpsFactory(
        op=ops.sin,
        ref=torch.sin,
        sample_inputs_func=dynamic_shape_sample_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_dynamic_shape_cmp()
    fact.grad_dynamic_shape_cmp()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
def test_f_sin_nd_bfloat16(mode):
    """
    Feature: elementwise sin op bfloat16.
    Description: compare across several shapes for bfloat16 inputs.
    Expectation: results are close within tolerance.
    """
    shapes = [
        (9, 9, 4),
        (8, 8, 4, 4, 3, 4),
        (5, 8, 8, 4, 5, 3),
    ]
    fact = ElementwiseOpsFactory(
        op=ops.sin,
        ref=torch.sin,
    )
    fact.set_context_mode(mode=mode)
    fact.test_elementwise_op_nd(shapes, ms.bfloat16)
