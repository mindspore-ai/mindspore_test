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
"""Test Tensor methods"""

from mindspore import context, ops, jit

from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_type_guard():
    """
    Feature: Test Tensor TypeGuard.
    Description: Calling x.squeeze() will add a TypeGuard on x.squeeze.__self__.
    Expectation: No guard checking failure, no exception, no graph break.
    """

    def fn(x):
        return ops.add(x.squeeze(), 1)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.arange(0, 4)
    o1 = fn(x)

    compiled_fn = jit(fn, capture_mode='bytecode')
    x = ops.arange(0, 4)  # It is a StubTensor
    o2 = compiled_fn(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_stub_tensor_load_method():
    """
    Feature: Test Tensor LOAD_METHOD.
    Description: Properly handle the self parameter in Tensor methods to avoid treating self as a constant.
    Expectation: no exception, no graph break.
    """

    def fn(x: StubTensor):
        y = x.permute((0, 2, 1))
        return y + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = ops.randn(2, 3, 4)  # It is a StubTensor
    o1 = fn(x1)

    compiled_fn = jit(fn, capture_mode='bytecode')
    o2 = compiled_fn(x1)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)

    x2 = ops.randn(2, 3, 4)  # It is a StubTensor
    o1 = fn(x2)

    compiled_fn = jit(fn, capture_mode='bytecode')
    o2 = compiled_fn(x2)

    assert_equal(o1, o2)
    # Should not treat argument x as a constant tensor, should reuse the first graph.
    assert_executed_by_graph_mode(fn, call_count=2)
