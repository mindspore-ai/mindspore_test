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

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.mint import nanmedian
from mindspore.ops import GradOperation, zeros_like

from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


# API support according to platforms, mode and jit level:
#
# 1. In Pynative mode, both nanmedian() and nanmedian(dim) is supported on all platforms;
# 2. In Graph mode, it can be described as a table:
#
# | device  | jit level | available interface |
# | :-----: | :-------: | :-----------------: |
# | Ascend  |    O0     | exclude Tensor(dim) |
# | Ascend  |    O2     |       nothing       |
# |   CPU   |   O0/O2   |     Tensor(dim)     |


def _np(x: Tensor) -> np.ndarray:
    np_type = np.float32 if x.dtype != ms.int64 else np.int64
    return x.asnumpy().astype(np_type)


def _equals(result: Tensor, exp: Tensor, ms_dtype):
    assert result.dtype == ms_dtype
    assert np.allclose(_np(result), _np(exp), equal_nan=True)


@test_utils.run_with_cell
def tensor_nanmedian(x, *args, **kwargs):
    return x.nanmedian(*args, **kwargs)


@test_utils.run_with_cell
def mint_nanmedian(x, *args, **kwargs):
    return nanmedian(x, *args, **kwargs)


class _Grad(nn.Cell):
    def __init__(self, net: nn.Cell, sens: Tensor):
        super().__init__()
        self.net = net
        self.grad_op = GradOperation(sens_param=True)
        self.grad_wrt_output = sens

    def construct(self, x: Tensor, *args) -> Tensor:
        sens = self.grad_wrt_output
        # nanmedian() and nanmedian(dim) requires different grad input
        sens = (sens, zeros_like(sens, dtype=ms.int64)) if args else sens
        return self.grad_op(self.net)(x, *args, sens)


def _tensor3x4(ms_dtype):
    """A tensor whose global=5, dim0=[4, 5, float('nan'), 7] with index=2(index[2] is uncertain), dim1=[1, 9, 5]T with index=1"""
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    x[:, 2] = float('nan')
    x[1:3] = x[2:0:-1]
    return Tensor(x, dtype=ms_dtype)


def _test_nanmedian_global(ms_dtype, f):
    _equals(f(_tensor3x4(ms_dtype)), Tensor(5, dtype=ms_dtype), ms_dtype)
    # grad
    grad_np = np.zeros((3, 4))
    grad_np[2, 1] = 42
    x = _tensor3x4(ms_dtype)
    dx = _Grad(f, Tensor(42, dtype=ms_dtype))(x)
    _equals(dx, Tensor(grad_np, dtype=ms_dtype), ms_dtype)


def _test_nanmedian_dim(ms_dtype, f):
    # dim0=[4, 5, float('nan'), 7] + index=2
    values, indices = f(_tensor3x4(ms_dtype), 0)  # positional invoke
    exp = Tensor([4, 5, float('nan'), 7], dtype=ms_dtype)
    _equals(values, exp, ms_dtype)
    assert indices.dtype == ms.int64
    # using self[indices] to judge: indices[2] depends on platform
    assert np.allclose(_np(_tensor3x4(ms_dtype))[_np(indices)][0], _np(exp), equal_nan=True)
    # dim1=[1, 9, 5] + index=1
    values, indices = f(_tensor3x4(ms_dtype), dim=1, keepdim=True)  # named invoke
    _equals(values, Tensor([[1], [9], [5]], dtype=ms_dtype), ms_dtype)
    _equals(indices, Tensor(np.ones((1, 3), dtype=np.int64)), ms.int64)
    # grad (using dim=1 and keepdim=False)
    dx = _Grad(f, Tensor(np.arange(1, 4), ms_dtype))(_tensor3x4(ms_dtype), 1, False)
    _equals(dx, Tensor([[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]], dtype=ms_dtype), ms_dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nanmedian_ascend(mode):
    """
    Feature: mint/Tensor ops.
    Description: test Tensor/mint.nanmedian() and Tensor/mint.nanmedian(dim, keepdim) on Ascend
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    _test_nanmedian_global(ms.float32, mint_nanmedian)
    _test_nanmedian_dim(ms.float32, mint_nanmedian)
    _test_nanmedian_global(ms.float32, tensor_nanmedian)
    if mode != ms.GRAPH_MODE:   # Tensor.nanmedian(dim) is unavailable on Ascend in Graph Mode
        _test_nanmedian_dim(ms.float32, tensor_nanmedian)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0', card_mark='onecard', essential_mark='essential'
)
@pytest.mark.parametrize(
    'mode, level',
    [
        (ms.GRAPH_MODE, 'O0'),
        (ms.PYNATIVE_MODE, 'PYNATIVE'),
        (ms.GRAPH_MODE, 'O2'),
    ]
)
def test_nanmedian_cpu(mode, level):
    """
    Feature: Tensor ops.
    Description: test Tensor.nanmedian(dim) on CPU with GRAPH O2 support, and Tensor.nanmedian() with all support
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level=level)
    else:
        # mint.nanmedian and Tensor.nanmedian() only available in Pynative mode
        _test_nanmedian_dim(ms.float32, mint_nanmedian)
        # nanmedian() on CPU has different grad result, skip it
        _test_nanmedian_global(ms.float32, mint_nanmedian)
        _test_nanmedian_global(ms.float32, tensor_nanmedian)

    # Tensor.nanmedian(dim) in all mode
    _test_nanmedian_dim(ms.float32, tensor_nanmedian)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nanmedian_bfloat16(mode):
    """
    Feature: mint&Tensor ops.
    Description: test mint.nn.GLU and mint.nn.functional.glu with bfloat16 support. (AICPU and CPU not supports)
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")

    _test_nanmedian_global(ms.bfloat16, mint_nanmedian)
    _test_nanmedian_dim(ms.bfloat16, mint_nanmedian)
    _test_nanmedian_global(ms.bfloat16, tensor_nanmedian)
    if mode != ms.GRAPH_MODE:   # Tensor.nanmedian(dim) is unavailable on Ascend in Graph Mode
        _test_nanmedian_dim(ms.bfloat16, tensor_nanmedian)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nanmedian_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test Tensor/mint.nanmedian() and Tensor/mint.nanmedian(dim, keepdim) dynamic shape feature.
    Expectation: expect correct result.
    """
    x1 = _tensor3x4(ms.float32)
    dim1 = 1

    x2_shape = (5, 5, 5)
    x2_np = np.random.random(x2_shape).astype(np.float32)
    x2_np.flat[np.random.choice(np.prod(x2_shape), 25, replace=False)] = np.nan
    x2 = Tensor(x2_np)
    dim2 = 2

    TEST_OP(tensor_nanmedian, [[x1.copy()], [x2.copy()]], 'nan_median', disable_mode=["GRAPH_MODE"])
    # nanmedian(dim) on Ascend in Graph Mode [O0] is available only when using mint
    TEST_OP(
        mint_nanmedian,
        [
            [x1, dim1, False],
            [x2, dim2, True],
        ],
        'nan_median_dim',
        disable_mode=["GRAPH_MODE"]
    )
