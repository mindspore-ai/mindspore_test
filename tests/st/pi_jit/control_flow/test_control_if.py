import numpy as np
from mindspore.nn import Cell
from mindspore.common import dtype as ms
from mindspore import Tensor
from mindspore import context, jit
import mindspore.ops.operations as op
from ..share.utils import match_array
from ..share.grad import GradOfAllInputs
from tests.mark_utils import arg_mark
from mindspore._c_expression import get_code_extra


class ControlOneIfOneAddnOneAddn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input1, input2):
        if x > y:
            out = self.addn([input1, input1, input1])
        else:
            out = self.addn([input2, input2, input2])
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_addn_addn_true():
    """
    Feature: PIJit
    Description: create a net, with if, True AddN input1
    Expectation: No exception.
    """
    x = Tensor(1, ms.float32)
    y = Tensor(0, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneIfOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddn.construct, capture_mode="ast")(ps_net, x, y, Tensor(input1), Tensor(input2))
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneIfOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddn.construct, capture_mode="bytecode")(pi_net, x, y, Tensor(input1), Tensor(input2))
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_addn_addn_false():
    """
    Feature: PIJit
    Description: create a net, with if, False AddN input2
    Expectation: No exception.
    """
    x = Tensor(0, ms.float32)
    y = Tensor(1, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneIfOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddn.construct, capture_mode="ast")(ps_net, x, y, Tensor(input1), Tensor(input2))
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneIfOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddn.construct, capture_mode="bytecode")(pi_net, x, y, Tensor(input1), Tensor(input2))
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])


class ControlOneIfOneAddnOneAddnOneAddn(Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input1, input2):
        if x > y:
            out = self.addn([input1, input1, input1])
        else:
            out = self.addn([input2, input2, input2])
        out_me = self.addn([out, input1])
        return out_me


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_addn_addn_addn_true():
    """
    Feature: PIJit
    Description: create a net, with if, True AddN input1, then Addn
    Expectation: No exception.
    """
    x = Tensor(1, ms.float32)
    y = Tensor(0, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneIfOneAddnOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddnOneAddn.construct, capture_mode="ast")(ps_net, x, y, Tensor(input1), Tensor(input2))
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneIfOneAddnOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddnOneAddn.construct, capture_mode="bytecode")(pi_net, x, y, Tensor(input1), Tensor(input2))
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ctrl_if_addn_addn_addn_false():
    """
    Feature: PIJit
    Description: create a net, with if, False AddN input2, then Addn
    Expectation: No exception.
    """
    x = Tensor(0, ms.float32)
    y = Tensor(1, ms.float32)
    input_shape = (1024, 512, 7, 7)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = ControlOneIfOneAddnOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddnOneAddn.construct, capture_mode="ast")(ps_net, x, y, Tensor(input1), Tensor(input2))
    ps_out = ps_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    ps_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = ControlOneIfOneAddnOneAddnOneAddn()
    jit(function=ControlOneIfOneAddnOneAddnOneAddn.construct, capture_mode="bytecode")(pi_net, x, y, Tensor(input1), Tensor(input2))
    pi_out = pi_net(x, y, Tensor(input1), Tensor(input2))
    grad_net = GradOfAllInputs(ps_net, sens_param=False)
    pi_grad = grad_net(x, y, Tensor(input1), Tensor(input2))
    match_array(ps_out, pi_out)
    match_array(ps_grad[2], pi_grad[2])
    match_array(ps_grad[3], pi_grad[3])

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jump_forward_if_not_none():
    """
    Feature: PIJit
    Description: test jump forward if not none
    Expectation: No exception.
    """
    def func(x, y):
        if x is None:
            return y + 1
        else:
            return y + 2

    x, y = None, Tensor(1)
    fn = jit(function=func, capture_mode="bytecode")
    got = fn(x, y)
    expected = func(x, y)
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["break_count_"] == 0

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jump_forward_if_none():
    """
    Feature: PIJit
    Description: test jump forward if none
    Expectation: No exception.
    """
    def func(x, y):
        if x is not None:
            return y + 2
        else:
            return y + 3

    x, y = 1, Tensor(1)
    fn = jit(function=func, capture_mode="bytecode")
    got = fn(x, y)
    expected = func(x, y)
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["break_count_"] == 0

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jump_backward_if_not_none():
    """
    Feature: PIJit
    Description: test jump backward if not none
    Expectation: No exception.
    """
    def func(x, y):
        while x is not None:
            x = None
        return y + 3

    x, y = 1, Tensor(1)
    fn = jit(function=func, capture_mode="bytecode")
    got = fn(x, y)
    expected = func(x, y)
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["break_count_"] == 0

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jump_backward_if_none():
    """
    Feature: PIJit
    Description: test jump backward if none
    Expectation: No exception.
    """
    def func(x, y):
        while x is None:
            x = 1
        return y + 4

    x, y = None, Tensor(1)
    fn = jit(function=func, capture_mode="bytecode")
    got = fn(x, y)
    expected = func(x, y)
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["break_count_"] == 0