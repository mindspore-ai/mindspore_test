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
import torch

import mindspore as ms
from mindspore import JitConfig, mint, mutable
from mindspore.nn import Cell
from mindspore.ops.auto_generate.gen_ops_def import add_ext as add
from mindspore.common.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.ops.share.add_sub_mint import AddSubMintOpFactory
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor, make_tensor_with_np_array
from tests.st.ops.share._op_info.op_info import op_db, ops_info, dtypes_extra_uint
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase

rtol = 1e-3


class AddCell(Cell):
    def __init__(self):
        super().__init__()
        self.add = add

    def construct(self, x, y, alpha):
        return self.add(x, y, alpha)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_forward(context_mode):
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()

    x = np.random.randn(1, 16, 4096, 128).astype(np.float32)
    y = np.random.randn(1, 16, 4096, 128).astype(np.float32)
    alpha = 2.0

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    add_cell.set_inputs(ms.tensor(shape=[None, None, None, None], dtype=ms.float16),
                        ms.tensor(shape=[None, None, None, None], dtype=ms.float16), alpha)

    x = np.random.randn(64, 20, 77, 77).astype(np.float16)
    y = np.random.randn(64, 1, 77, 77).astype(np.float16)

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    add_cell.set_inputs(ms.tensor(shape=[None, None, None, None], dtype=ms.float16),
                        ms.tensor(shape=None, dtype=ms.float16), alpha)

    x = np.random.randn(3, 73, 3, 768).astype(np.float16)
    y = np.random.randn(1).astype(np.float16)

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_ops_dynamic():
    """
    Feature: ops.extend.add
    Description: dynamic shape and rank
    Expectation: success
    """
    x1 = ms.Tensor(np.array([[1, 2], [3, 4]], np.float32))
    y1 = ms.Tensor(np.array([[5, 6], [7, 8]], np.float32))
    x2 = ms.Tensor(np.array([[1, 2, 3]], np.float32))
    y2 = ms.Tensor(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], np.float32))

    TEST_OP(add, [[x1, y1, 1.], [x2, y2, 2.]], case_config={'disable_input_check': True, 'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_backward(context_mode):
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()

    # 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[5, 6], [7, 8]], np.float32)
    alpha = 2.0

    output = ms.grad(add_cell, (0))(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = np.ones_like(y)

    np.testing.assert_allclose(output, expect, rtol=rtol)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bf16(context_mode):
    """
    Feature: ops.extend.add
    Description: bf16
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()

    # 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[5, 6], [7, 8]], np.float32)
    alpha = 2.0

    output = ms.grad(add_cell, (0))(ms.tensor(x, ms.bfloat16), ms.tensor(y, ms.bfloat16), alpha).float().asnumpy()
    expect = np.ones_like(y)

    np.testing.assert_allclose(output, expect, rtol=rtol)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bool(context_mode):
    """
    Feature: test add backward
    Description: test add backward
    Expectation: success
    """
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()
    add_cell.set_jit_config(JitConfig(jit_level='O0'))

    # 2 x 2
    x = np.array([[True, True], [False, False]], np.bool_)
    y = np.array([[True, False], [True, False]], np.bool_)
    alpha = True

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)


def ops_add_binary_compare(input_binary_data, output_binary_data):
    add_cell = AddCell()
    output = add_cell(ms.Tensor(input_binary_data[0]), ms.Tensor(input_binary_data[1]), 1.0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = ms.grad(add_cell, (0))(ms.Tensor(input_binary_data[0]), ms.Tensor(input_binary_data[1]), 1.0)
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)


@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                output_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((84, 144, 32), np.float32), ((84, 144, 32), np.float32)],
                                output_info=[((84, 144, 32), np.float32), ((84, 144, 32), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1024,), np.float32), ((), np.float32)],
                                output_info=[((1024,), np.float32), ((1024,), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((48, 32, 32), np.float32), ((), np.float32)],
                                output_info=[((48, 32, 32), np.float32), ((48, 32, 32), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case4(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 6, 288, 64), np.float32), ((), np.float32)],
                                output_info=[((1, 6, 288, 64), np.float32), ((1, 6, 288, 64), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case5(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 1, 1, 288, 64), np.float32), ((), np.float32)],
                                output_info=[((1, 1, 1, 288, 64), np.float32), ((1, 1, 1, 288, 64), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case6(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 576, 128, 16, 2), np.float32), ((1, 576, 128, 16, 1), np.float32)],
                                output_info=[((1, 576, 128, 16, 2), np.float32), ((1, 576, 128, 16, 2), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case7(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_binary_cases(mode):
    """
    Feature: Ops
    Description: test op add
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)

    ops_add_binary_case1()
    ops_add_binary_case2()
    ops_add_binary_case3()
    ops_add_binary_case4()
    ops_add_binary_case5()
    ops_add_binary_case6()
    ops_add_binary_case7()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
@ops_info(op_db['add_ext'])
def test_mint_f_add_nd_same_dtype(mode, op_info):
    '''
    Feature: mint.add
    Description: Same-shape nd tensors; compare forward and gradients.
    Expectation: MindSpore matches PyTorch for outputs and gradients.
    '''
    fact = AddSubMintOpFactory(
        op_info=op_info,
        op_kwargs=dict(alpha=2),
    )
    fact.set_context_mode(mode=mode)
    fact.test_binary_op_nd_same_dtype()
    fact.test_binary_op_nd_same_dtype(disable_op_info_dtypes=[ms.complex64, ms.complex128], broad_cast=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
@ops_info(op_db['add_ext'])
def test_mint_f_add_nd_same_dtype_without_bfloat16(mode, op_info):
    '''
    Feature: mint.add
    Description: Same-shape nd tensors without bfloat16; compare forward and gradients.
    Expectation: MindSpore matches PyTorch for outputs and gradients.
    '''
    fact = AddSubMintOpFactory(
        op_info=op_info,
        op_kwargs=dict(alpha=2),
    )
    fact.set_context_mode(mode=mode)
    fact.test_binary_op_nd_same_dtype(disable_op_info_dtypes=[ms.bfloat16])
    fact.test_binary_op_nd_same_dtype(
        disable_op_info_dtypes=[ms.bfloat16, ms.complex64, ms.complex128],
        broad_cast=True
    )


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
@ops_info(op_db['add_ext'])
def test_mint_f_add_nd_same_extra_dtype(mode, op_info):
    '''
    Feature: mint.add
    Description: Extra unsigned integer dtypes (uint16/uint32/uint64) combinations; forward-only.
    Expectation: MindSpore forward matches PyTorch.
    '''
    fact = AddSubMintOpFactory(
        op_info=op_info,
        op_kwargs=dict(alpha=2),
    )
    fact.set_context_mode(mode=mode)
    fact.test_binary_op_nd_same_dtype(dtypes=dtypes_extra_uint)
    fact.test_binary_op_nd_same_dtype(dtypes=dtypes_extra_uint, broad_cast=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_other_not_tensor(mode):
    '''
    Feature: mint.add
    Description: 'other' is not a Tensor; validate type checking.
    Expectation: Raises TypeError.
    '''
    input_x = make_tensor((5,), ms.float32)
    other = (1,)
    alpha = -3
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
        op_input=input_x,
        op_args=(other,),
        op_kwargs=dict(alpha=alpha),
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
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_other_shape_not_match(mode):
    '''
    Feature: mint.add
    Description: Shape mismatch between inputs (non-broadcastable).
    Expectation: Raises ValueError.
    '''
    input_x = make_tensor((5,), ms.float32)
    other = make_tensor((6,), ms.float32)
    alpha = -3
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
        op_input=input_x,
        op_args=(other,),
        op_kwargs=dict(alpha=alpha),
    )
    fact.set_context_mode(mode=mode)
    with pytest.raises(ValueError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative'])
def test_mint_f_add_float32_2d_discontinuous_tensor(mode):
    '''
    Feature: mint.add
    Description: Non-contiguous float32 input tensors; test forward and gradients.
    Expectation: Outputs and gradients match PyTorch.
    '''
    input_x = make_tensor((9, 4), ms.float32, discontiguous=True)
    other = make_tensor((9, 4), ms.float32, discontiguous=True)
    alpha = 3
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
        op_input=input_x,
        op_args=(other,),
        op_kwargs=dict(alpha=alpha),
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
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_float32_inf_nan_broadcast(mode):
    '''
    Feature: mint.add
    Description: float32 inputs containing NaN/Inf with broadcast; verify numerical
    propagation; compare forward and gradients.
    Expectation: Outputs and gradients match PyTorch (NaN/Inf propagate consistently).
    '''
    def mint_add_inf_nan_broadcast_sample_inputs_func():
        sample_inputs = [
            OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((3,), np.nan), ms.float32),
                op_args=(make_tensor_with_np_array(np.full((1,), np.nan), ms.float32),),
                op_kwargs=dict(alpha=12.3),
                op_name='mint_add_nan_broadcast_float32',
            ),
            OpSampleInput(
                op_input=make_tensor_with_np_array(np.full((8, 5, 4), np.inf), ms.float32),
                op_args=(make_tensor_with_np_array(np.full((8, 5, 1), np.inf), ms.float32),),
                op_kwargs=dict(alpha=12.3),
                op_name='mint_add_inf_broadcast_float32',
            ),
        ]
        return sample_inputs

    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
        sample_inputs_func=mint_add_inf_nan_broadcast_sample_inputs_func,
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
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_mixed_dtype_forward(mode):
    '''
    Feature: mint.add
    Description: Mixed dtypes (bool/int/float) combinations; forward-only comparison.
    Expectation: MindSpore forward matches PyTorch.
    '''
    normal_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.float16, ms.float32, ms.float64]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_mixed_dtype(normal_dtypes, normal_dtypes, grad_cmp=False)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_mixed_dtype_backward(mode):
    '''
    Feature: mint.add
    Description: Mixed float dtypes; gradient comparison.
    Expectation: Gradients match PyTorch.
    '''
    grad_dtypes = [ms.float16, ms.float32, ms.float64]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_mixed_dtype(grad_dtypes, grad_dtypes, grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_mixed_dtype_input_bfloat16_forward(mode):
    '''
    Feature: mint.add
    Description: One input bfloat16 with other dtypes; forward-only comparison.
    Expectation: MindSpore forward matches PyTorch.
    '''
    normal_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,
                     ms.float16, ms.float32, ms.float64, ms.bfloat16]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_mixed_dtype([ms.bfloat16], normal_dtypes, grad_cmp=False)
    fact.test_add_sub_mixed_dtype(normal_dtypes, [ms.bfloat16], grad_cmp=False)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_mixed_dtype_input_bfloat16_backward(mode):
    '''
    Feature: mint.add
    Description: One input bfloat16 with float dtypes; gradient comparison.
    Expectation: Gradients match PyTorch.
    '''
    grad_dtypes = [ms.float16, ms.float32, ms.float64, ms.bfloat16]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_mixed_dtype([ms.bfloat16], grad_dtypes, grad_cmp=True)
    fact.test_add_sub_mixed_dtype(grad_dtypes, [ms.bfloat16], grad_cmp=True)


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_mixed_extra_dtype_forward(mode):
    '''
    Feature: mint.add
    Description: Extra unsigned integer dtypes (uint16/uint32/uint64) combinations; forward-only.
    Expectation: MindSpore forward matches PyTorch.
    '''
    extra_dtypes = [ms.uint16, ms.uint32, ms.uint64]
    mixed_dtypes = [ms.bool,]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_mixed_dtype(extra_dtypes, mixed_dtypes, grad_cmp=False)
    fact.test_add_sub_mixed_dtype(mixed_dtypes, extra_dtypes, grad_cmp=False)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_scalar_mixed_tensor_forword(mode):
    '''
    Feature: mint.add
    Description: Scalar with tensor (both positions) across dtypes; forward-only.
    Expectation: MindSpore forward matches PyTorch.
    '''
    normal_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64,
                     ms.float16, ms.float32, ms.float64]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_scalar_tensor_mixed(6, normal_dtypes, scalar_is_input=True, grad_cmp=False)
    fact.test_add_sub_scalar_tensor_mixed(3.0, normal_dtypes, scalar_is_input=False, grad_cmp=False)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_scalar_mixed_tensor_backword(mode):
    '''
    Feature: mint.add
    Description: Scalar with tensor (both positions) for float dtypes; gradient comparison.
    Expectation: Gradients match PyTorch.
    '''
    normal_dtypes = [ms.float16, ms.float32, ms.float64]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_scalar_tensor_mixed(2.33, normal_dtypes, scalar_is_input=True, grad_cmp=True)
    fact.test_add_sub_scalar_tensor_mixed(6, normal_dtypes, scalar_is_input=False, grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_scalar_mixed_tensor_bfloat16(mode):
    '''
    Feature: mint.add
    Description: Scalar with bfloat16 tensor (both positions); gradient comparison.
    Expectation: Gradients match PyTorch.
    '''
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_scalar_tensor_mixed(1.2, [ms.bfloat16], scalar_is_input=True, grad_cmp=True)
    fact.test_add_sub_scalar_tensor_mixed(7, [ms.bfloat16], scalar_is_input=False, grad_cmp=True)


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_scalar_mixed_tensor_extra_dtype_forword(mode):
    '''
    Feature: mint.add
    Description: Scalar with extra unsigned integer dtypes; forward-only.
    Expectation: MindSpore forward matches PyTorch.
    '''
    extra_dtypes = [ms.uint16, ms.uint32, ms.uint64]
    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
    )
    fact.set_context_mode(mode=mode)
    fact.test_add_sub_scalar_tensor_mixed(True, extra_dtypes, scalar_is_input=True, grad_cmp=False)
    fact.test_add_sub_scalar_tensor_mixed(False, extra_dtypes, scalar_is_input=False, grad_cmp=False)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_dynamic_shape(mode):
    '''
    Feature: mint.add
    Description: Dynamic shape with fixed rank (5D), keeping alpha as mutable;
    compare forward and gradients across concrete shapes.
    Expectation: Outputs and gradients match PyTorch.
    '''
    def add_dynamic_shape_sample_inputs_func():
        sample_inputs = []
        compile_input = OpSampleInput(
            op_input=ms.Tensor(shape=(None, None, None, None, None), dtype=ms.float32),
            op_args=(ms.Tensor(shape=(None, None, None, 1, None), dtype=ms.float32),),
            op_kwargs=dict(alpha=mutable(input_data=3.3, dynamic_len=False)),
            op_name='add_compile_input'
        )
        sample_inputs.append(compile_input)
        op_params = [
            ((5, 5, 8, 5, 4), (5, 5, 8, 1, 4), mutable(input_data=4.3, dynamic_len=False)),
            ((9, 9, 8, 8, 4), (9, 9, 8, 1, 4), mutable(input_data=-2.1, dynamic_len=False)),
        ]
        for input_shape, other_shape, alpha in op_params:
            sample_inputs.append(OpSampleInput(
                op_input=make_tensor(input_shape, ms.float32),
                op_args=(make_tensor(other_shape, ms.float32),),
                op_kwargs=dict(alpha=alpha),
                op_name='add_running_input'
            ))
        return sample_inputs

    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
        sample_inputs_func=add_dynamic_shape_sample_inputs_func,
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
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_mint_f_add_dynamic_rank(mode):
    '''
    Feature: mint.add
    Description: Dynamic rank (shape=None) for inputs, keeping alpha as mutable;
    compare forward and gradients across ranks.
    Expectation: Outputs and gradients match PyTorch.
    '''
    def add_dynamic_shape_rank_inputs_func():
        sample_inputs = []
        compile_input = OpSampleInput(
            op_input=ms.Tensor(shape=None, dtype=ms.float32),
            op_args=(ms.Tensor(shape=None, dtype=ms.float32),),
            op_kwargs=dict(alpha=mutable(input_data=2.33, dynamic_len=False)),
            op_name='add_compile_input'
        )
        sample_inputs.append(compile_input)
        op_params = [
            ((5, 5), (5, 1), mutable(input_data=9.6, dynamic_len=False)),
            ((9, 9, 7), (9, 9, 7), mutable(input_data=10.10, dynamic_len=False)),
        ]
        for input_shape, other_shape, alpha in op_params:
            sample_inputs.append(OpSampleInput(
                op_input=make_tensor(input_shape, ms.float32),
                op_args=(make_tensor(other_shape, ms.float32),),
                op_kwargs=dict(alpha=alpha),
                op_name='add_running_input'
            ))
        return sample_inputs

    fact = AddSubMintOpFactory(
        op=mint.add,
        ref=torch.add,
        sample_inputs_func=add_dynamic_shape_rank_inputs_func,
    )
    fact.set_context_mode(mode=mode)
    fact.forward_dynamic_shape_cmp()
    fact.grad_dynamic_shape_cmp()
