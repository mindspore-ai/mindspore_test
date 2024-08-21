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
from tests.st.compiler.control.cases_register import case_register
import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, context, ops, nn
from mindspore.common import dtype as mstype


context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_ascend
@case_register.target_gpu
def test_while_loop():
    """
    Feature: control flow
    Description: Using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: No exception.
    """

    def complex_pure_function(init_value):
        input_tensor, init = init_value
        activation = ops.ReLU()
        fc = activation(input_tensor)
        init = init + 1
        return [fc, init]

    def cond_func(init_value):
        return init_value[-1] < 100

    @jit
    def test_while_loop_inner(init_val):
        whileop = ops.WhileLoop()
        result = whileop(cond_func, complex_pure_function, init_val)
        return result

    dtype = mstype.float32
    input_tensor = Tensor(np.random.rand(6, 6), dtype)
    init = 0
    init_state = [input_tensor, init]
    result = test_while_loop_inner(init_state)
    assert result[-1] == 100


@case_register.level1
@case_register.target_ascend
@case_register.target_gpu
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


@case_register.level0
@case_register.target_ascend
@case_register.target_gpu
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

    @jit
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


@case_register.level1
@case_register.target_ascend
@case_register.target_gpu
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

    @jit
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


@case_register.level0
@case_register.target_ascend
@case_register.target_gpu
def test_foriloop_unroll():
    """
    Feature: control flow
    Description: Using ForiLoopEvaluator to handle ops.ForiLoop operation
    Expectation: No exception.
    """

    def complex_pure_function(index, val):
        add = ops.Add()
        return add(val, index)

    @jit
    def test_fori_loop_inner(result_init):
        fori_loop = ops.ForiLoop()
        return fori_loop(0, 10, complex_pure_function, result_init)

    result_init = Tensor(0, mstype.float32)
    result = test_fori_loop_inner(result_init)
    print(result)
    assert result == 45


@case_register.skip(reason="Cannot process ops.xx both in loop_func and whileloop declaration in construct")
@case_register.level1
@case_register.target_ascend
@case_register.target_gpu
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


@case_register.skip(reason="Unsupported loop func with side effect")
@case_register.level1
@case_register.target_ascend
@case_register.target_gpu
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

    @jit
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
