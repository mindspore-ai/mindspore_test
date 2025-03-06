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
""" test jit syntax level """

import os
import pytest
import mindspore as ms
from mindspore import nn, context
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


jit_config_strict = ms.JitConfig(jit_syntax_level="STRICT")
jit_config_compatible = ms.JitConfig(jit_syntax_level="COMPATIBLE")
jit_config_lax = ms.JitConfig(jit_syntax_level="LAX")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_in_cell():
    """
    Feature: JIT Fallback
    Description: Test jit_syntax_level for nn.cell.
    Expectation: No exception.
    """
    context.set_context(jit_level='O0')
    class Net(nn.Cell):
        def construct(self):
            return {"a": 1}

    os.unsetenv("MS_DEV_JIT_SYNTAX_LEVEL")
    net1 = Net()
    net1.set_jit_config(jit_config_strict)
    out1 = net1()
    assert isinstance(out1, tuple)

    net2 = Net()
    net2.set_jit_config(jit_config_compatible)
    out2 = net2()
    assert isinstance(out2, dict)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_in_function():
    """
    Feature: JIT Fallback
    Description: Test jit_syntax_level for function decorated by @jit.
    Expectation: No exception.
    """
    def func():
        return {"a": 1}

    os.unsetenv("MS_DEV_JIT_SYNTAX_LEVEL")
    func1 = ms.jit(function=func, fullgraph=True)
    out1 = func1()
    assert isinstance(out1, tuple)

    func2 = ms.jit(function=func)
    out2 = func2()
    assert isinstance(out2, dict)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_class_in_cell():
    """
    Feature: JIT Fallback
    Description: Test jit_syntax_level for nn.cell.
    Expectation: No exception.
    """
    class InnerNet:
        def __init__(self):
            self.number = 2

        def func(self, x):
            return self.number * x.asnumpy()

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.cls = InnerNet()

        def construct(self, x):
            return self.cls.func(x)

    os.unsetenv("MS_DEV_JIT_SYNTAX_LEVEL")
    # The jit_syntax_level is LAX here due to ms.context.
    ms.set_context(jit_syntax_level=ms.LAX)
    x = ms.Tensor(2)
    net1 = Net()
    assert net1(x) == 4

    # JitConfig will override the jit_syntax_level of ms.context.
    with pytest.raises(AttributeError):
        net2 = Net()
        net2.set_jit_config(jit_config_compatible)
        net2(x)

    # Environment variable 'MS_DEV_JIT_SYNTAX_LEVEL' has the highest priority.
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    with pytest.raises(AttributeError):
        net3 = Net()
        net3.set_jit_config(jit_config_lax)
        net3(x)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_class_in_function():
    """
    Feature: JIT Fallback
    Description: Test jit_syntax_level for function decorated by @jit.
    Expectation: No exception.
    """
    class InnerNet:
        def __init__(self):
            self.number = 2

        def func(self, x):
            return self.number * x.asnumpy()

    cls = InnerNet()

    def func(x):
        return cls.func(x)

    reserved_env = os.getenv('MS_DEV_JIT_SYNTAX_LEVEL')
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'
    # The jit_syntax_level is LAX here due to env.
    func1 = ms.jit(function=func, fullgraph=True)
    x = ms.Tensor(2)
    assert func1(x) == 4

    # Environment variable 'MS_DEV_JIT_SYNTAX_LEVEL' has the highest priority.
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    with pytest.raises(AttributeError):
        func3 = ms.jit(function=func)
        func3(x)

    if reserved_env is None:
        os.unsetenv('MS_DEV_JIT_SYNTAX_LEVEL')
    else:
        os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = reserved_env


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_class_in_function_with_deprecated_jit():
    """
    Feature: JIT Fallback
    Description: Test jit_syntax_level for function decorated by deprecated @jit.
    Expectation: No exception.
    """
    import mindspore._deprecated.jit as jit

    class InnerNet:
        def __init__(self):
            self.number = 2

        def func(self, x):
            return self.number * x.asnumpy()

    cls = InnerNet()

    def func(x):
        return cls.func(x)

    x = ms.Tensor(2)
    with pytest.raises(AttributeError):
        func1 = jit(fn=func, jit_config=jit_config_strict)
        func1(x)

    func2 = jit(fn=func, jit_config=jit_config_lax)
    x = ms.Tensor(2)
    assert func2(x) == 4
