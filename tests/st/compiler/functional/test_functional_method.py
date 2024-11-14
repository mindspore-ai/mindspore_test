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
"""Test the feature of functional"""
import pytest
import numpy as np
import mindspore as ms
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_method_clamp():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min, max):  # pylint: disable=redefined-builtin
        return x.clamp(min, max)

    x = ms.Tensor([1, 2, 3, 4, 5])
    out_clamp_tensor = func(x, ms.Tensor(2), ms.Tensor(4))
    out_clamp_scalar = func(x, 2, 4)
    expect = ms.Tensor([2, 2, 3, 4, 4])
    assert np.all(out_clamp_tensor.asnumpy() == expect.asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_default():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min):  # pylint: disable=redefined-builtin
        return x.clamp(min)

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    expect = ms.Tensor([2, 2, 3, 4, 5])
    assert np.all(out.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_any():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min, max):  # pylint: disable=redefined-builtin
        return x.clamp(min, ms.Tensor(max.asnumpy()))

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2), ms.Tensor(4))
    expect = ms.Tensor([2, 2, 3, 4, 4])
    assert np.all(out.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min):  # pylint: disable=redefined-builtin
        return x.clamp(min, max=ms.Tensor(4)), x.clamp(max=3, min=1)

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([1, 2, 3, 3, 3]).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_keyword_any():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit
    def func(x, min, max):  # pylint: disable=redefined-builtin
        return x.clamp(min=ms.Tensor(min.asnumpy())), x.clamp(2, max=int(max.asnumpy()))

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2), ms.Tensor(4))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 5]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_exception():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Raise expected exception.
    """
    @ms.jit
    def func(x, min, max):  # pylint: disable=redefined-builtin
        return x.clamp(min, max)

    with pytest.raises(TypeError) as raise_info:
        func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2), 4)
    assert "Failed calling clamp with" in str(raise_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @ms.jit
    def func(x):
        return x.max()

    x = ms.Tensor([1, 2, 3, 4, 5])
    assert func(x).asnumpy() == ms.Tensor(5).asnumpy()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @ms.jit
    def func(x):
        return x.max(keepdims=False)

    x = ms.Tensor([1, 2, 3, 4, 5])
    assert func(x).asnumpy() == ms.Tensor(5).asnumpy()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_max_keyword_any():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.max.
    Expectation: Run success
    """
    @ms.jit
    def func(x, y):
        return x.max(return_indices=bool(y.asnumpy()))

    x = ms.Tensor([1, 2, 3, 4, 5])
    y = ms.Tensor(False)
    assert func(x, y).asnumpy() == ms.Tensor(5).asnumpy()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_reshape():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.reshape.
    Expectation: Run success
    """
    @ms.jit
    def func(x):
        return x.reshape(1, 2, 3)

    x = ms.Tensor([[[1], [2]], [[3], [4]], [[5], [6]]])
    out = func(x)
    expect = ms.Tensor([[[1, 2, 3], [4, 5, 6]]])
    assert np.all(out.asnumpy() == expect.asnumpy())
