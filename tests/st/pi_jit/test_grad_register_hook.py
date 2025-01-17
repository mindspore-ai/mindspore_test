import pytest
from mindspore import jit, context, Tensor, Parameter, ops, ParameterTuple,JitConfig
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore._c_expression import get_code_extra
from .share.utils import match_array
from tests.mark_utils import arg_mark

def hook_1_fn(grad):
    return grad * 2

def hook_2_fn(grad):
    return grad * 4

def hook_3_fn(grad):
    return grad * 6

def hook_4_fn(grad):
    return grad * 8

def break_hook_fn(grad):
    print("grad : ", grad, flush=True)
    return grad * 20

class ForwardNet(Cell):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.matmul = ops.MatMul()
        self.w = Parameter(Tensor([[2., 2.], [2., 2.]], mstype.float32), name="w", requires_grad=True)
        self.z = Parameter(Tensor([[3., 3.], [3., 3.]], mstype.float32), name="z", requires_grad=True)

    def construct(self, x, y):
        x = x * self.w * self.z
        return self.matmul(x, y)

class GradNet(Cell):
    def __init__(self, get_all, get_by_list):
        super(GradNet, self).__init__()
        self.net = ForwardNet()
        self.params = ParameterTuple(self.net.trainable_params())
        self.params[0].register_hook(hook_3_fn)
        self.params[1].register_hook(hook_4_fn)
        self.get_by_list = get_by_list
        self.grad_op = ops.GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, x, y):
        if self.get_by_list is False:
            return self.grad_op(self.net)(x, y)
        return self.grad_op(self.net, self.params)(x, y)

def run_grad_net(net, x, y):
    return net(x, y)

def run_multi_grad_net(net, x, y):
    a = net(x, y)
    b = net(x, y)
    return a, b

def check_func_compile_state(func):
    jcr = get_code_extra(func.__wrapped__)
    assert jcr is not None
    assert jcr['break_count_'] == 0
    assert jcr['stat'] == 'GRAPH_CALLABLE'

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_first_input(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_first_input_multi_hook(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input_multi_hook with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    x.register_hook(y_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(True, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_all_inputs(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_all_inputs with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, True)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_only_weights(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_only_weights with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(True, True)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_inputs_and_weights(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_inputs_and_weights with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_multi_grad_net])
@pytest.mark.parametrize('net', [GradNet(True, True)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_multi_grad_inputs_and_weights(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_multi_grad_inputs_and_weights with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.skip(reason='fix it later')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [break_hook_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_first_input_break(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input_break with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(net, x, y)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
