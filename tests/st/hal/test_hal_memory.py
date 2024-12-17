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
import mindspore.context as context
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.device_utils import set_device

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Abs()

    def construct(self, x):
        return self.ops(x)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_memory_stats():
    """
    Feature: runtime memory api.
    Description: Test runtime.memory_stats api.
    Expectation: runtime.memory_stats api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.runtime.memory_stats()
    _pynative_executor.sync()
    assert not res is None
    assert isinstance(res, dict)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_memory_reserved():
    """
    Feature: runtime memory api.
    Description: Test runtime.memory_reserved api.
    Expectation: runtime.memory_reserved api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.runtime.memory_reserved()
    _pynative_executor.sync()
    assert not res is None
    assert isinstance(res, int)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_memory_allocated():
    """
    Feature: runtime memory api.
    Description: Test runtime.memory_allocated api.
    Expectation: runtime.memory_allocated api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.runtime.memory_allocated()
    _pynative_executor.sync()
    assert not res is None
    assert isinstance(res, int)

@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_max_memory_reserved():
    """
    Feature: runtime memory api.
    Description: Test runtime.max_memory_reserved api.
    Expectation: runtime.max_memory_reserved api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.runtime.max_memory_reserved()
    _pynative_executor.sync()
    assert not res is None
    assert isinstance(res, int)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_max_memory_allocated():
    """
    Feature: runtime memory api.
    Description: Test runtime.max_memory_allocated api.
    Expectation: runtime.max_memory_allocated api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.runtime.max_memory_allocated()
    _pynative_executor.sync()
    assert not res is None
    assert isinstance(res, int)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_memory_summary():
    """
    Feature: runtime memory api.
    Description: Test runtime.memory_summar api.
    Expectation: runtime.memory_summar api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.runtime.memory_summary()
    _pynative_executor.sync()
    assert not res is None
    assert isinstance(res, str)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_reset_peak_memory_stats():
    """
    Feature: runtime memory api.
    Description: Test runtime.reset_peak_memory_stats api.
    Expectation: runtime.reset_peak_memory_stats api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    ms.runtime.reset_peak_memory_stats()
    res = ms.runtime.memory_stats()
    _pynative_executor.sync()
    assert (res["max_reserved_memory"] == 0 and res["max_allocated_memory"] == 0)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_reset_max_memory_reserved():
    """
    Feature: runtime memory api.
    Description: Test runtime.reset_max_memory_reserved api.
    Expectation: runtime.reset_max_memory_reserved api performs as expected in grad.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)

    net = Net()
    net(Tensor(2.0))
    ms.runtime.reset_max_memory_reserved()
    res = ms.runtime.max_memory_reserved()
    _pynative_executor.sync()
    assert res == 0


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_reset_max_memory_allocated():
    """
    Feature: runtime memory api.
    Description: Test runtime.reset_max_memory_allocated api.
    Expectation: runtime.reset_max_memory_allocated api performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    ms.runtime.reset_max_memory_allocated()
    res = ms.runtime.max_memory_allocated()
    _pynative_executor.sync()
    assert res == 0
