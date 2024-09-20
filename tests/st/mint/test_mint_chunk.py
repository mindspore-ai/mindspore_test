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
import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import mint, Tensor, jit, context, JitConfig, ops
from mindspore.common.api import _pynative_executor


# from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

@test_utils.run_with_cell
def chunk_forward_func(x, chunks, dim):
    return mint.chunk(x, chunks, dim)


@test_utils.run_with_cell
def chunk_backward_func(x, chunks, dim):
    return ops.grad(chunk_forward_func, (0,))(x, chunks, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ['GE', 'pynative', 'KBK'])
def test_chunk_foward_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op Chunk.
    Expectation: expect correct result.
    """
    #forward
    np_x = np.array(np.arange(10).reshape((5, 2)), dtype=np.float32)
    x = ms.Tensor(np_x, dtype=ms.float32)
    dims = 0
    chunks = 3
    expect = [np.array(np.arange(4).reshape((2, 2)), dtype=np.float32),
              np.array(np.arange(4, 8).reshape((2, 2)), dtype=np.float32),
              np.array(np.arange(8, 10).reshape((1, 2)), dtype=np.float32)]
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        out = chunk_forward_func(x, chunks, dims)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        out = (jit(chunk_forward_func, jit_config=JitConfig(jit_level="O0")))(x, chunks, dims)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = chunk_forward_func(x, chunks, dims)
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

    #backward
    x = Tensor(np.arange(20).reshape(10, 2), dtype=ms.float32)
    chunks = 2
    expect_grad = np.ones((10, 2))
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        grad = chunk_backward_func(x, chunks, 0)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        grad = (jit(chunk_backward_func, jit_config=JitConfig(jit_level="O0")))(x, chunks, 0)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        grad = chunk_backward_func(x, chunks, 0)
    assert np.allclose(grad.asnumpy(), expect_grad)
    assert grad.asnumpy().shape == x.shape


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_forward_dynamic_shape(context_mode):
    """
    Feature: chunk ops.
    Description: test ops chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    input_dyn = Tensor(shape=[4, None, None], dtype=ms.int64)
    chunks = 3
    dims = 0
    test_cell = test_utils.to_cell_obj(mint.chunk)
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(60).reshape((4, 3, 5)).astype(np.int64))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = [np.array(np.arange(30).reshape((2, 3, 5)), dtype=np.float32),
                     np.array(np.arange(30, 60).reshape((2, 3, 5)), dtype=np.float32)]
    for res, exp in zip(out, expect_output):
        assert np.allclose(res.asnumpy(), exp)

    input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.int64))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = [np.array(np.arange(12).reshape((2, 2, 3)), dtype=np.float32),
                     np.array(np.arange(12, 24).reshape((2, 2, 3)), dtype=np.float32)]
    for res, exp in zip(out, expect_output):
        assert np.allclose(res.asnumpy(), exp)

    if context_mode == ms.GRAPH_MODE:
        dims = 2
        test_cell.set_inputs(input_dyn, chunks, dims)
        input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.int64))
        with pytest.raises(RuntimeError):
            _ = test_cell(input_tensor, chunks, dims)
            _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
def test_chunk_forward_dynamic_rank(context_mode):
    """
    Feature: chunk ops.
    Description: test ops chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    input_dyn = Tensor(shape=None, dtype=ms.int64)
    chunks = 3
    dims = 0
    test_cell = test_utils.to_cell_obj(mint.chunk)
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.int64))
    with pytest.raises(RuntimeError):
        _ = test_cell(input_tensor, chunks, dims)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_backward_dynamic_shape(context_mode):
    """
    Feature: chunk ops.
    Description: test ops chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    input_dyn = Tensor(shape=[None, 4, None], dtype=ms.float32)
    chunks = 3
    dims = 1
    test_cell = test_utils.to_cell_obj(ops.grad(mint.chunk, (0,)))
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(60).reshape((3, 4, 5)).astype(np.float32))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = np.ones((3, 4, 5))
    assert np.allclose(out.asnumpy(), expect_output)

    input_tensor = Tensor(np.arange(24).reshape((2, 4, 3)).astype(np.float32))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = np.ones((2, 4, 3))
    assert np.allclose(out.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
def test_chunk_backward_dynamic_rank(context_mode):
    """
    Feature: chunk ops.
    Description: test ops chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    input_dyn = Tensor(shape=None, dtype=ms.float64)
    chunks = 3
    dims = 1
    test_cell = test_utils.to_cell_obj(ops.grad(mint.chunk, (0,)))
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.float64))
    with pytest.raises(RuntimeError):
        _ = test_cell(input_tensor, chunks, dims)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_forward_mutable(context_mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op Chunk.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)
    x = Tensor(np.arange(20).reshape(10, 2), dtype=ms.float32)
    chunks = 2
    dims = 0
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    if context_mode == ms.GRAPH_MODE:
        with pytest.raises(RuntimeError):
            _ = chunk_forward_func(x, ms.mutable(chunks), dims)
            _pynative_executor.sync()

        with pytest.raises(RuntimeError):
            _ = chunk_forward_func(x, chunks, ms.mutable(dims))
    else:
        out = chunk_forward_func(x, ms.mutable(chunks), ms.mutable(dims))
        for res, exp in zip(out, expect):
            assert np.allclose(res.asnumpy(), exp)
            _pynative_executor.sync()
