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
"""Test basic operation with one stage"""
import sys
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype
from mindspore.common.api import jit
from tests.mark_utils import arg_mark
from mindspore._c_expression import get_code_extra

@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping tests on Python 3.11 and higher.")

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "compile_by_trace": True,
    "print_bb": False,
    "MAX_INLINE_DEPTH": 10,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_tuple():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return (x, x+1, x+2)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_list():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return [x, x+1, x+2]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_return_add_result_tuple():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x + y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = (1, 2, 3)
    b = (4, 5, 6)
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a, b)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == (1, 2, 3, 4, 5, 6)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_return_add_result_list():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x + y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = [1, 2, 3]
    b = [4, 5, 6]
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a, b)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == [1, 2, 3, 4, 5, 6]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_tuple_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = Parameter(Tensor([1, 2, 3]))
            self.b = Parameter(Tensor([1, 1, 1]))

        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, ())
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_list_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = Parameter(Tensor([1, 2, 3]))
            self.b = Parameter(Tensor([1, 1, 1]))

        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, [])
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_dict_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = Parameter(Tensor([1, 2, 3]))
            self.b = Parameter(Tensor([1, 1, 1]))

        def construct(self, x):
            return self.a + self.b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, {})
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_slice():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = (x, x+1, x+2)
            return m[0:2:1]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[0:2:1]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice_with_default_parameter():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[0:2]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 2
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice_with_default_parameter_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[::]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_slice_with_default_parameter_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = [x, x+1, x+2]
            return m[:]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, list)
    assert len(ret) == 3
    assert ret[0] == Tensor([1])
    assert ret[1] == Tensor([2])
    assert ret[2] == Tensor([3])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_dict():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {"x": x, "y": x+1}
            return m["x"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == Tensor([1])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_dict_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {}
            m["x"] = x
            return m["x"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == Tensor([1])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_dict_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            m = {"x": x+1}
            return m["x"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = Tensor([1])
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x/y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = (1.0, 2.0, 3.0)
    b = Tensor(np.ones([2, 3]).astype(np.float32))
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a, b)
    expect = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(ret.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_input():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return x/y

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = [1.0, 2.0, 3.0]
    b = Tensor(np.ones([2, 3]).astype(np.float32))
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, a, b)
    expect = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(ret.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_constant():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            a, b = x
            return (a, b)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    m = (1, 2)
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, m)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == (1, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_constant_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            a, b = x
            return (a, b)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    m = [1, 2]
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, m)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == (1, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_mutable_kwargs_args():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, a, *args, b=1, **kwargs):
            return a + b + args[0] + kwargs["s"]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, 1, 10, 100, s=1000)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == 1012


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_handle_mutable_kwargs_args_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, a, *args, b=1, **kwargs):
            return a + b + args[0]

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret = jit(Net.construct, mode="PIJit", jit_config=cfg)(net, 1, 10, 100, s=1000)
    jcr = get_code_extra(Net.construct)
    assert jcr["break_count_"] == 0
    assert ret == 12


@pytest.mark.skip(reason="fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_free_variable():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit", jit_config={"loop_unrolling": True, "compile_by_trace": True})
        def construct(self, x):
            mod = 2
            return any(i % mod == 0 for i in x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    input1 = (1, 2, 3, 4)
    assert net(input1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    input2 = (1, 1, 1, 1, 1)
    assert not net(input2)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@pytest.mark.skip(reason="When disable loop_unrolling, check guard failed.")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_use_free_variable_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit", jit_config={"compile_by_trace": True})
        def construct(self, x):
            mod = 2
            return any(i % mod == 0 for i in x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    input1 = (1, 2, 3, 4)
    assert net(input1)
    input2 = (1, 1, 1, 1, 1)
    assert not net(input2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_getattr():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = 1

        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret1 = net(1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    net.a = 2
    ret2 = net(2)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert ret1 == 2
    assert ret2 == 4


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_getattr_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.a = 1

        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return self.a + x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    ret1 = net(1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    net.a = 2
    ret2 = net(1)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert ret1 == 2
    assert ret2 == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cycle_container_structure():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = [1, 2]
    a += [a]
    ret = net(a)
    assert ret == a


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cycle_container_structure_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = {"1": 1}
    a["2"] = a
    ret = net(a)
    assert ret == a


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cycle_container_structure_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit", jit_config=cfg)
        def construct(self, x):
            return x

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    a = [1, 2, 3]
    b = [4, 5, 6]
    a[0] = b
    b[0] = a
    ret1 = net(a)
    assert ret1 == a
    ret2 = net(b)
    assert ret2 == b


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_guard_parameter():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor(np.random.rand(2, 2), dtype.float32), name='w')

        @jit(mode="PIJit")
        def construct(self, x):
            return self.w * x

    context.set_context(mode=context.PYNATIVE_MODE)
    m = Tensor([[1, 1], [2, 2]], dtype.float32)
    net1 = Net()
    ret1 = net1(m)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    net2 = Net()
    ret2 = net2(m)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert np.allclose(ret1.asnumpy(), (net1.w * m).asnumpy())
    assert np.allclose(ret2.asnumpy(), (net2.w * m).asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_dict_items_call_in_control_flow():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, x, y, z):
            m = {"1": x + z, "2": y - z}
            ret = 0
            for _, v in m.items():
                ret = ret + v
            return ret

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    c = Tensor([2, 2, 2])
    net = Net()
    ret = net(a, b, c)
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_tensor_method_by_ast():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, x):
            return x.view((2, 2))

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3, 4])
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_tensor_method_by_ast_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, x):
            return x.var()

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3, 4])
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_tensor_method_by_ast_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, x):
            return x.get("1")

    context.set_context(mode=context.PYNATIVE_MODE)
    a = {"1": Tensor([1, 2, 3, 4])}
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_tensor_method_by_ast_4():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, x):
            return x.contiguous()

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3, 4])
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_function_parse_by_ast():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, x):
            return ops.split(x, [3, ])

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor(np.ones((3, 128, 352, 224)))
    net = Net()
    net(a)
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_constant_flod_for_variable():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        @jit(mode="PIJit")
        def construct(self, x, y):
            if all(x > y):
                out = x + y
            else:
                out = x * y
            return out

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([3, 4, 5])
    b = Tensor([1, 2, 3])
    net = Net()
    ret = net(a, b)
    assert np.all(ret.asnumpy() == np.array([4, 6, 8]))
