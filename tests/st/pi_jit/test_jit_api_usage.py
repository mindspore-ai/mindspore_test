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
"""Test jit api usage"""

import numpy as np

import mindspore as ms
from mindspore import Tensor

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_equal


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_jit_wrap_cell_class():
    """
    Feature: ms.jit(class) usage.
    Description: Apply ms.jit with capture_mode='bytecode' as a class decorator for nn.Cell.
    Expectation: JIT compiled class produces the same result as pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_cell_class
    """

    class PopNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = -1

        def construct(self, items):
            return items.pop(self.idx)

    pynative_net = PopNet()
    pynative_result = pynative_net([1, 2, 3])

    JitPopNet = ms.jit(PopNet, capture_mode="bytecode")
    jit_net = JitPopNet()
    jit_result = jit_net([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_jit_wrap_cell_instance():
    """
    Feature: ms.jit(Cell instance) usage.
    Description: Wrap an nn.Cell instance with ms.jit using capture_mode='bytecode'.
    Expectation: JIT wrapped instance matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_cell_instance
    """

    class PopNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = -1

        def construct(self, items):
            return items.pop(self.idx)

    pynative_net = PopNet()
    pynative_result = pynative_net([1, 2, 3])

    jit_net = ms.jit(pynative_net, capture_mode="bytecode")
    jit_result = jit_net([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_jit_wrap_net_method():
    """
    Feature: ms.jit(method) usage.
    Description: Wrap the construct method of nn.Cell with ms.jit capture_mode='bytecode'.
    Expectation: JIT wrapped method matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_net_method
    """

    class PopNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = -1

        def construct(self, items):
            return items.pop(self.idx)

    net = PopNet()
    pynative_result = net([1, 2, 3])

    jit_construct = ms.jit(net.construct, capture_mode="bytecode")
    jit_result = jit_construct([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_jit_wrap_callable_object():
    """
    Feature: ms.jit(callable object) usage.
    Description: Wrap a callable object with ms.jit capture_mode='bytecode'.
    Expectation: JIT wrapped callable matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_wrap_other_instance
    """

    class PopCallable:
        def __init__(self):
            self.idx = -1

        def __call__(self, items):
            return items.pop(self.idx)

    callable_obj = PopCallable()
    pynative_result = callable_obj([1, 2, 3])

    jit_callable = ms.jit(callable_obj, capture_mode="bytecode")
    jit_result = jit_callable([1, 2, 3])

    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_jit_with_custom_config():
    """
    Feature: ms.jit configuration.
    Description: Use ms.jit with capture_mode='bytecode' and jit_level='O0' inside nn.Cell.
    Expectation: JIT compiled network matches pynative execution.
    Migrated from: test_pijit_use.py::test_pijit_jit_config
    """

    @ms.jit(capture_mode="bytecode", jit_level="O0")
    def double_tensor(x):
        return x + x

    class ConfigNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 1

        def construct(self, x):
            mid = double_tensor(x) * self.scale
            return mid + mid

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    tensor_input = Tensor(input_np)

    pynative_net = ConfigNet()
    pynative_result = pynative_net(tensor_input)

    jit_net = ConfigNet()
    jit_net.construct = ms.jit(jit_net.construct, capture_mode="bytecode")
    jit_result = jit_net(tensor_input)

    assert_equal(pynative_result, jit_result)
