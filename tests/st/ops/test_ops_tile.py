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

"""Test Tile."""

from functools import reduce
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, ops
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})


def tile_func(x, multiplies):
    return ops.operations.manually_defined.tile(x, multiplies)


@test_utils.run_with_cell
def tile_forward_func(x, multiplies):
    return tile_func(x, multiplies)


@test_utils.run_with_cell
def tile_backward_func(x, multiplies):
    return ops.grad(tile_forward_func, (0,))(x, multiplies)  # pylint: disable=not-callable


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_forward(mode):
    """
    Feature: Ops.
    Description: test op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_x1 = np.random.rand(3, 4, 5, 6).astype(np.float32)
    x1 = Tensor(np_x1)
    mul1 = (2, 2, 2, 2)
    out1 = tile_forward_func(x1, mul1)
    expect1 = np.tile(np_x1, mul1)
    assert np.allclose(out1.asnumpy(), expect1)

    np_x2 = np.random.rand(3, 4).astype(np.float32)
    x2 = Tensor(np_x2)
    mul2 = (2, 2, 2, 2)
    out2 = tile_forward_func(x2, mul2)
    expect2 = np.tile(np_x2, mul2)
    assert np.allclose(out2.asnumpy(), expect2)

    np_x3 = np.random.rand(3, 4, 5, 6).astype(np.float32)
    x3 = Tensor(np_x3)
    mul3 = (2,)
    out3 = tile_forward_func(x3, mul3)
    expect3 = np.tile(np_x3, mul3)
    assert np.allclose(out3.asnumpy(), expect3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x1 = Tensor(np.random.rand(3, 4, 5, 6).astype(np.float32))
    mul1 = (2, 2, 2, 2)
    grads1 = tile_backward_func(x1, ms.mutable(mul1))
    expect1 = np.ones((3, 4, 5, 6)).astype(np.float32) * reduce(lambda x, y: x * y, mul1)
    assert np.allclose(grads1.asnumpy(), expect1)

    x2 = Tensor(np.random.rand(3, 4).astype(np.float32))
    mul2 = (2, 2, 2, 2)
    grads2 = tile_backward_func(x2, ms.mutable(mul2))
    expect2 = np.ones((3, 4)).astype(np.float32) * reduce(lambda x, y: x * y, mul2)
    assert np.allclose(grads2.asnumpy(), expect2)

    x3 = Tensor(np.random.rand(3, 4, 5, 6).astype(np.float32))
    mul3 = (2,)
    grads3 = tile_backward_func(x3, ms.mutable(mul3))
    expect3 = np.ones((3, 4, 5, 6)).astype(np.float32) * reduce(lambda x, y: x * y, mul3)
    assert np.allclose(grads3.asnumpy(), expect3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_vmap(mode):
    """
    Feature: test vmap function.
    Description: test tile op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (-1, None)
    np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    mul = (1, 1, 2, 2)
    x = Tensor(np.tile(np_x.reshape((2, 3, 4, 5, 1, 1)), (1, 1, 1, 1, 2, 2)))
    nest_vmap = ops.vmap(ops.vmap(tile_forward_func, in_axes=in_axes), in_axes=in_axes)
    out = nest_vmap(x, mul)
    expect = np.tile(np.tile(np_x, mul).reshape(1, 1, 2, 3, 8, 10), (2, 2, 1, 1, 1, 1))
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_tile_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op concat.
    Expectation: expect tile result.
    """
    ms.context.set_context(runtime_num_threads=1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.rand(3, 4, 5, 6).astype(np.float32))
    input_case2 = Tensor(np.random.rand(3, 4).astype(np.float32))
    TEST_OP(tile_func, [[input_case1, (2, 3, 2, 3)], [input_case2, (3, 2, 3, 2)]], 'tile', disable_input_check=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_tile_forward_dyn(mode, dyn_mode):
    """
    Feature: Ops.
    Description: test op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    mul = (1, 1, 2, 2)
    dyn_tensor_shape = [None, None, None, None] if dyn_mode == "dyn_shape" else None
    dyn_x = Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    fwd_cell = test_utils.to_cell_obj(tile_func)
    fwd_cell.set_inputs(dyn_x, ms.mutable(mul))

    shape1 = (2, 3, 4, 5)
    np_x1 = np.random.rand(*shape1).astype(np.float32)
    x1 = Tensor(np_x1)
    out1 = fwd_cell(x1, ms.mutable(mul))
    expect1 = np.tile(np_x1, mul)
    assert np.allclose(out1.asnumpy(), expect1)

    shape2 = (3, 4, 5, 6) if dyn_mode == "dyn_shape" else (2, 3, 4)
    np_x2 = np.random.rand(*shape2).astype(np.float32)
    x2 = Tensor(np_x2)
    out2 = fwd_cell(x2, ms.mutable(mul))
    expect2 = np.tile(np_x2, mul)
    assert np.allclose(out2.asnumpy(), expect2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE])  # ms.context.GRAPH_MODE has runtime heterogeneous bug.
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_tile_backward_dyn(mode, dyn_mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    mul = (1, 1, 2, 2)
    dyn_tensor_shape = [None, None, None, None] if dyn_mode == "dyn_shape" else None
    dyn_x = Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    bwd_cell = test_utils.to_cell_obj(tile_backward_func)
    bwd_cell.set_inputs(dyn_x, ms.mutable(mul))

    shape1 = (2, 3, 4, 5)
    x1 = Tensor(np.random.rand(*shape1).astype(np.float32))
    grads1 = bwd_cell(x1, mul)
    expect1 = np.ones((2, 3, 4, 5)).astype(np.float32) * 4.0
    assert np.allclose(grads1.asnumpy(), expect1)

    shape2 = (3, 4, 5, 6) if dyn_mode == "dyn_shape" else (2, 3, 4)
    x2 = Tensor(np.random.rand(*shape2).astype(np.float32))
    grads2 = bwd_cell(x2, mul)
    expect2 = np.ones(shape2).astype(np.float32) * 4.0
    assert np.allclose(grads2.asnumpy(), expect2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize('is_fwd', [True, False])
def test_tile_dynamic_len(mode, is_fwd):
    """
    Feature: test dynamic len.
    Description: test op tile.
    Expectation: expect correct result.
    """
    if mode == ms.context.GRAPH_MODE and not is_fwd:
        # ms.context.GRAPH_MODE and backward has runtime heterogeneous bug.
        return
    ms.context.set_context(mode=mode)
    np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    x = Tensor(np_x)
    mul = (1, 1, 2, 2)

    if is_fwd:
        # Forward.
        forward_cell = test_utils.to_cell_obj(tile_func)
        forward_cell.set_inputs(x, ms.mutable(mul, True))
        out = forward_cell(x, ms.mutable(mul))
        fwd_expect = np.tile(np_x, mul)
        assert np.allclose(out.asnumpy(), fwd_expect)
    else:
        # Backward.
        backward_cell = test_utils.to_cell_obj(tile_backward_func)
        backward_cell.set_inputs(x, ms.mutable(mul, True))
        grads = backward_cell(x, ms.mutable(mul))
        bwd_expect = np.ones((2, 3, 4, 5)).astype(np.float32) * 4.0
        assert np.allclose(grads.asnumpy(), bwd_expect)


def ops_tile_binary_case_compare(input_binary_data, output_binary_data, dims):
    output = tile_forward_func(Tensor(input_binary_data[0]), dims)
    assert np.allclose(output.asnumpy(), output_binary_data[0])
    grads = tile_backward_func(Tensor(input_binary_data[0]), dims)
    assert np.allclose(grads.asnumpy(), output_binary_data[1])


@ops_binary_cases(OpsBinaryCase(input_info=[((3, 32), np.float32)],
                                output_info=[((1, 3, 81920), np.float32), ((3, 32), np.float32)],
                                extra_info='SDv1'))
def ops_tile_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_tile_binary_case_compare(input_binary_data, output_binary_data, (1, 1, 2560))


@ops_binary_cases(OpsBinaryCase(input_info=[((3, 32), np.float32)],
                                output_info=[((1, 3, 327680), np.float32), ((3, 32), np.float32)],
                                extra_info='SDv1'))
def ops_tile_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_tile_binary_case_compare(input_binary_data, output_binary_data, (1, 1, 10240))


@ops_binary_cases(OpsBinaryCase(input_info=[((3, 32), np.float32)],
                                output_info=[((1, 3, 655360), np.float32), ((3, 32), np.float32)],
                                extra_info='SDv1'))
def ops_tile_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_tile_binary_case_compare(input_binary_data, output_binary_data, (1, 1, 20480))


@ops_binary_cases(OpsBinaryCase(input_info=[((3, 32), np.float32)],
                                output_info=[((1, 3, 1310720), np.float32), ((3, 32), np.float32)],
                                extra_info='SDv1'))
def ops_tile_binary_case4(input_binary_data=None, output_binary_data=None):
    ops_tile_binary_case_compare(input_binary_data, output_binary_data, (1, 1, 40960))


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 32), np.float32)],
                                output_info=[((1, 64, 114688), np.float32), ((64, 32), np.float32)],
                                extra_info='SD5B'))
def ops_tile_binary_case5(input_binary_data=None, output_binary_data=None):
    ops_tile_binary_case_compare(input_binary_data, output_binary_data, (1, 1, 3584))


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 32), np.float32)],
                                output_info=[((1, 64, 1835008), np.float32), ((64, 32), np.float32)],
                                extra_info='SD5B'))
def ops_tile_binary_case6(input_binary_data=None, output_binary_data=None):
    ops_tile_binary_case_compare(input_binary_data, output_binary_data, (1, 1, 57344))


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_binary_cases(mode):
    """
    Feature: Ops
    Description: test op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)

    ops_tile_binary_case1()
    ops_tile_binary_case2()
    ops_tile_binary_case3()
    ops_tile_binary_case4()
    ops_tile_binary_case5()
    ops_tile_binary_case6()
