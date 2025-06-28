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
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, jit, context, ops, nn
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)
context.set_context(jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_while_loop():
    """
    Feature: control flow
    Description: Using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: No exception.
    """

    def complex_pure_function(init_value):
        input_tensor = init_value
        activation = ops.ReLU()
        fc = activation(input_tensor)
        return  fc + 1

    def cond_func(init_value):
        return init_value.value() < 100

    @jit(backend="ms_backend")
    def test_while_loop_inner(init_val):
        whileop = ops.WhileLoop()
        result = whileop(cond_func, complex_pure_function, init_val)
        return result

    dtype = mstype.float32
    input_tensor = ms.Parameter(Tensor([1], dtype))
    init_state = input_tensor
    result = test_while_loop_inner(init_state)
    assert result == 100


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_while_loop2():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.whileop = ops.WhileLoop()
            self.activation = ops.ReLU()

        def construct(self, result_init):
            def complex_pure_function(init_value):
                input_tensor, init = init_value
                fc = self.activation(input_tensor)
                init = init + 1
                return [fc, init]

            def cond_func(init_value):
                return init_value[-1] < 100

            result = self.whileop(cond_func, complex_pure_function, result_init)
            return result

    dtype = mstype.float32
    input_tensor = Tensor(np.random.rand(6, 6), dtype)
    init = 0
    init_state = [input_tensor, init]
    net = Net()
    result = net(init_state)
    assert result[-1] == 100


@arg_mark(plat_marks=['platform_gpu',], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_while_loop3():
    """
    Feature: control flow
    Description: Using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: No exception.
    """

    def cond_func(init_value):
        return init_value[1] > 1

    def while_function(init_value):
        input_tensor, init, add = init_value
        out = add(input_tensor, init)
        init = init - 1
        return [out, init, add]

    class WhileLoopNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.whileop = ops.WhileLoop()

        def construct(self, inputs):
            out = inputs
            res = self.whileop(cond_func, while_function, [out, 3, self.add])
            out = res[0]
            return out

    net = WhileLoopNet()
    out = net(Tensor([2]))
    assert out == 7


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_scan_unroll():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def complex_pure_function(input_tensor, el):
        activation = ops.ReLU()
        fc = activation(input_tensor)
        return fc, el

    @jit(backend="ms_backend")
    def test_scan_inner(result_init, array):
        scan_op = ops.Scan()
        return scan_op(complex_pure_function, result_init, array, len(array), True)

    array = []
    result_init = Tensor(np.random.rand(6, 6), mstype.float32)
    i = 0
    while i < 10:
        input_tensor = Tensor(np.random.rand(6, 6), mstype.float32)
        array.append(input_tensor)
        i = i + 1
    result = test_scan_inner(result_init, array)
    assert len(result[-1]) == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_scan_not_unroll():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def complex_pure_function(input_tensor, el):
        activation = ops.ReLU()
        fc = activation(input_tensor)
        return fc, el

    @jit(backend="ms_backend")
    def test_scan_inner(result_init, array):
        scan_op = ops.Scan()
        return scan_op(complex_pure_function, result_init, array, len(array), False)

    array = []
    result_init = Tensor(np.random.rand(6, 6), mstype.float32)
    i = 0
    while i < 10:
        input_tensor = Tensor(np.random.rand(6, 6), mstype.float32)
        array.append(input_tensor)
        i = i + 1
    result = test_scan_inner(result_init, array)
    assert len(result[-1]) == 10


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_scan_simple_loop():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def simple_loop_func(res, el):
        res = res + el
        return res, res

    @jit(backend="ms_backend")
    def test_simple_scan_inner(result_init):
        array = [1, 2, 3, 4]
        scan_op = ops.Scan()
        return scan_op(simple_loop_func, result_init, array, len(array), False)

    result_init = ms.Tensor(0)
    result = test_simple_scan_inner(result_init)
    assert result == (10, [1, 3, 6, 10])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_foriloop_unroll():
    """
    Feature: control flow
    Description: Using ForiLoopEvaluator to handle ops.ForiLoop operation
    Expectation: No exception.
    """

    def complex_pure_function(index, val):
        add = ops.Add()
        return add(val, index)

    @jit(backend="ms_backend")
    def test_fori_loop_inner(result_init):
        fori_loop = ops.ForiLoop()
        return fori_loop(0, 10, complex_pure_function, result_init)

    result_init = Tensor(0, mstype.float32)
    result = test_fori_loop_inner(result_init)
    print(result)
    assert result == 45


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_high_order_with_unroll_as_false():
    """
    Feature: control flow
    Description: test higher order grad with unroll as false
    Expectation: Raise error with unsupported reason.
    """
    def for_in_foriloop_function(index, input_tensor):
        add = ops.Add()
        out = add(input_tensor, 1)
        for _ in range(3):
            out = add(out, index)
        return out

    class ForiLoopForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.fori_loop = ops.ForiLoop()

        def construct(self, inputs):
            out = inputs
            out = self.fori_loop(0, 7, for_in_foriloop_function, out, False)
            return out

    @jit
    def get_grad(x):
        net_2 = ForiLoopForNet()
        grad_net_2_f = ops.grad(net_2) # pylint: disable=E1102
        grad_net_2_s = ops.grad(grad_net_2_f) # pylint: disable=E1102
        return grad_net_2_s(x)

    with pytest.raises(RuntimeError, match="Loop op with unroll set as false is not allow do higher order grad"):
        x = Tensor(np.random.randn(32, 1).astype(np.float32))
        get_grad(x)


@pytest.mark.skip(reason="Cannot process ops.xx both in loop_func and whileloop declaration in construct")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_while_loop_unsupport1():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, result_init):

            def complex_pure_function(init_value):
                input_tensor, init = init_value
                activation = ops.ReLU()
                fc = activation(input_tensor)
                init = init + 1
                return [fc, init]

            def cond_func(init_value):
                return init_value[-1] < 100

            whileop = ops.WhileLoop()
            result = whileop(cond_func, complex_pure_function, result_init)
            return result

    dtype = mstype.float32
    input_tensor = Tensor(np.random.rand(6, 6), dtype)
    init = 0
    init_state = [input_tensor, init]
    net = Net()
    result = net(init_state)
    assert result[-1] == 100


@pytest.mark.skip(reason="Unsupported loop func with side effect")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_scan_unsupport1():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def complex_pure_function(input_tensor, el):
        activation = ops.ReLU()
        assign_op = ops.Assign()
        assign_op(input_tensor, activation(input_tensor))
        return input_tensor, input_tensor

    @jit(backend="ms_backend")
    def test_scan_inner(result_init, array):
        scan_op = ops.Scan()
        return scan_op(complex_pure_function, result_init, array, len(array), True)

    array = []
    result_init = ms.Parameter(Tensor(np.random.rand(6, 6), mstype.float32))
    i = 0
    while i < 10:
        input_tensor = Tensor(np.random.rand(6, 6), mstype.float32)
        array.append(input_tensor)
        i = i + 1
    result = test_scan_inner(result_init, array)
    assert len(result[-1]) == 10
