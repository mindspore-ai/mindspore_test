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
"""Test msadapter Tensor monkey-patch methods"""

import mindspore
from mindspore import jit, Tensor, ops

from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark


def size(self, dim=None):
    if dim is None:
        return self.shape
    assert isinstance(dim, int), f'`dim` must be int but got {type(dim)}'
    return self.shape[dim]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_msadapter_Tensor_size_monkey_patch():
    """
    Feature: Test msadapter Tensor size monkey patch method.
    Description: monkey patch Tensor.size with custom size() function.
    Expectation: no exception, no graph break.
    """

    # Monkey patch
    Tensor.size = size

    def fn(x: Tensor):
        a = x.size(0)
        b = x.size()[1]
        return x + a + b

    x = mindspore.tensor([[1, 2, 3, 4]])  # Shape is (1, 4)

    o1 = fn(x)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


def new_ones(self, *size, dtype=None):
    if dtype is None:
        dtype = self.dtype
    if isinstance(size[0], tuple):
        size = size[0]
    return ops.ones(size, dtype=dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_msadapter_Tensor_new_ones_monkey_patch():
    """
    Feature: Test msadapter Tensor.new_ones monkey patch method.
    Description: monkey patch Tensor.new_ones with custom new_ones() function.
    Expectation: no exception, no graph break.
    """

    # Monkey patch
    # The original Tensor.new_ones method is a functional overload op.
    Tensor.new_ones = new_ones

    def fn(x: Tensor):
        return x.new_ones(1, 2, dtype=mindspore.float32) + x

    x = mindspore.tensor([[1, 2], [3, 4]])  # Shape is (2, 2)

    o1 = fn(x)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)
