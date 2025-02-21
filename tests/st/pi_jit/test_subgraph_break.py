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
"""test graph break in call_function"""

import sys
import pytest

import mindspore as ms
from mindspore import Tensor, jit, context, ops, nn
from mindspore._c_expression import get_code_extra

from tests.st.pi_jit.share.utils import match_array, assert_has_graph_break, assert_equal,pi_jit_with_config
from tests.mark_utils import arg_mark

SKIP_PY37 = pytest.mark.skipif(sys.version_info[:2] == (3,7), reason="Not support py37 setup loop bytecode")

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False, 'subgraph_break_opt': True}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v1():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a * 2

    def f1(x: Tensor):
        y = f2(x)
        return y + 1

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v2():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a * 2

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v3():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        d = {'k': 2}  # alive local, unsupported output
        print('GRAPH BREAK', flush=True)  # break
        return a * d['k']

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v4():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        d = {'k': 2}  # alive local, unsupported output
        print('GRAPH BREAK', flush=True)  # break
        return a * d['k']

    def f1(x: Tensor):
        y = x * 2  # alive local
        d = {'bias': 1}  # alive local, unsupported output
        z = f2(x)
        return y + z + d['bias']

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_three_layers_v1():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f3.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f3(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b

    def f2(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f3(x)  # break
        return a + b + c

    def f1(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f2(x)
        return a + b + c

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_three_layers_v2():
    """
    Feature: test graph break in call_function.
    Description: two graph breaks in f3.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f3(x: Tensor):
        a = x + 1  # alive local
        print('GRAPH BREAK', flush=True)  # break
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b

    def f2(x: Tensor):
        a = x + 1  # alive local
        c = f3(x)  # break
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b + c

    def f1(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f2(x)
        return a + b + c

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_four_layers_v1():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f4.
    Expectation: The result of PIJit is same as pynative.
    """

    def f4(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b

    def f3(x: Tensor):
        a = f4(x)  # break
        return x - a

    def f2(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        return f3(a * b)  # break

    def f1(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f2(x)
        return a + b + c

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v1():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        for i in range(5):
            if i % 2 == 0:
                x += 1
            else:
                print('GRAPH BREAK', flush=True)  # break
        return x

    def f1(x: Tensor):
        a = f2(x)
        return a + x

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v2():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x, y):
        result = ops.zeros_like(x)
        for i in range(5):
            print('GRAPH BREAK', flush=True)  # graph break
            result += x * y
        return result

    def f2(x):
        y = ms.tensor([2.0])
        result = f3(x, y)
        return result * 2

    def f1(x):
        result = f2(x)
        return result + 1

    x = ops.randn(3, 3)
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v3():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, num_loops: int) -> Tensor:
        result = ops.zeros_like(x)
        result += num_loops
        for i in range(num_loops):
            result += x * i
            print('GRAPH BREAK', end='\n\n')
            result = ops.relu(result)
            result = result / (i + 1)
        return result

    def f2(x: Tensor, num_loops: int) -> Tensor:
        x = x * 2
        x = ops.sin(x)
        return f3(x, num_loops)

    def f1(x: Tensor, num_loops: int) -> Tensor:
        x = x + 1
        x = ops.cos(x)
        return f2(x, num_loops)

    x = ops.randn(10)
    num_loops = 5
    o1 = f1(x, num_loops)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, num_loops)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v4():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(input_tensor, loop_count):
        result = ops.zeros_like(input_tensor)
        for i in range(loop_count):
            temp = input_tensor * i
            result = result + ops.sin(temp)
            if i % 2 == 0:
                print('GRAPH BREAK', flush=True)  # graph break
                result = result * 2
            else:
                result = result / 2
                print('GRAPH BREAK', flush=True)  # graph break
        return result

    def f2(input_tensor, loop_count, factor):
        a = input_tensor + factor
        b = a * 2
        c = ops.relu(b)
        result = f3(c, loop_count)
        d = result - factor
        e = ops.sum(d)
        return e

    def f1(input_tensor, loop_count=5):
        factor = ms.tensor(2.0)
        intermediate_result = f2(input_tensor, loop_count, factor)  # 调用中间层函数
        final_result = intermediate_result * input_tensor
        output = ops.mean(final_result)
        return output

    x = ops.randn(3, 4)
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v5():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, repeat_times: int) -> Tensor:
        result = x
        for i in range(repeat_times):
            result = result * 2 + ops.sin(result)

        count = 0
        while count < 3:
            result = ops.relu(result) + ops.mean(result)
            count += 1
            print('GRAPH BREAK', flush=True)  # graph break
        return result

    def f2(input_dict: dict) -> Tensor:
        tensor_data = input_dict['data']
        repeat_count = input_dict['count']
        processed = tensor_data * 3.0 + ops.ones_like(tensor_data)
        return f3(processed, repeat_count)

    def f1(input_tensor: Tensor) -> Tensor:
        input_dict = {
            'data': input_tensor + 1.0,
            'count': 5
        }
        result = f2(input_dict)
        return ops.relu(result)

    x = ops.randn(3, 4)
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v6():
    """
    Feature: test graph break in call_function.
    Description: graph break at loop condition.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x: Tensor, y: int):
        a = ops.ones((5, 5))
        b = x + a
        c = ops.zeros((5, 5))

        while ops.sum(b) < 100:  # graph break
            b = b * 2.5
            c = c + b
            d = ops.matmul(b, c)

        e = ops.sin(d)
        return e.mean()

    def f1(t: Tensor, params: dict):
        x = t * 2
        y = params['factor']
        z = f2(x, y)
        w = ops.exp(z)
        v = w + ms.tensor(params['offset'])
        return ops.tanh(v)

    x = ops.ones((5, 5))
    params = {'factor': 3, 'offset': 1.5}
    o1 = f1(x, params)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, params)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v7():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at loop body; 2.f1 call f2 at last statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        count = 0
        while count < 3:
            print('GRAPH BREAK', end='\n\n')
            z = z * 2
            z = z - 1
            count += 1
        return z

    def f1(a, b):
        c = a ** 2
        d = b + 2
        return f2(c, d)

    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v8():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at loop body; 2.f1 call f2 at last statement, and f2 call f3 at last statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x, y):
        i = 0
        while i < y:
            i += 1
            print('GRAPH BREAK', end='\n\n')
            x = ops.sin(x)
        return x

    def f2(a, b):
        a = a + b
        a = ops.relu(a)
        return f3(a, b)

    def f1(x):
        x = x * 2.0
        x = ops.sqrt(x)
        return f2(x, 3)

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v9():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break in loop body, at break statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        i = 0
        while i < 10:
            z = z * 2
            if i % 2 != 0:
                print('GRAPH BREAK', end='\n\n')
                break
            z = z - 1
            i += 1
        return z

    def f1(a, b):
        c = a * b
        d = f2(c, b)
        e = d / a
        return e

    x = Tensor([1., 2., 3.])
    y = Tensor([4., 5., 6.])
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v10():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at else statement (after loop body).
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, threshold):
        result = ops.zeros_like(x)
        i = 0
        while i < x.shape[0]:
            result += x * 2
            if i > threshold:
                break  # shouldn't reach here
            result += 1
            i += 1
        else:
            print('GRAPH BREAK', end='\n\n')
            result = result / 2
        return result

    def f1(x, y):
        z = x + y
        w = f2(z, 10)
        return w * 2

    x = ops.randn(2, 4)
    y = ops.randn(2, 4)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v11():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at while-loop body; 2.the last statement of loop body is Return (which means this the loop will only be executed once)
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        i = 0
        while i < 3:
            z = z * 2
            print('GRAPH BREAK', end='\n\n')
            i += 1
            return z  # the last statement of loop body is Return

    def f1(a, b):
        c = a * b
        d = f2(c, b)
        e = d + c
        return e

    x = ops.randn(2, 4)
    y = ops.randn(2, 4)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v12():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at for-in-range-loop body; 2.the last statement of loop body is Return (which means this the loop will only be executed once)
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        for i in range(3):
            z = z + i
            print('GRAPH BREAK', end='\n\n')
            return z  # the last statement of loop body is Return

    def f1(a, b):
        c = a * b
        d = f2(c, b)
        e = d + c
        return e

    x = ops.randn(2, 4)
    y = ops.randn(2, 4)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v13():
    """
    Feature: test graph break in call_function.
    Description: 1.f1 call f2 in loop; 2.f2 has graph break.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x - y
        z = z * 2
        print('GRAPH BREAK', end='\n\n')
        z = ops.sin(z)
        return z

    def f1(x, params):
        result = x
        for i in range(3):
            result = ops.relu(result)
            result = f2(result, params['y'])
            result = ops.exp(result)
        return result

    x = Tensor([1., 2., 3.])
    params = {
        'y': Tensor([4.0, 5.0, 6.0])
    }
    o1 = f1(x, params)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, params)

    match_array(o1, o2, error=7)
    jcr = get_code_extra(f1)
    assert jcr['stat'] == 'NEVER_COMPILE'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_recursion():
    """
    Feature: test graph break in call_function.
    Description: graph break in recursion.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor, n: int):
        if n == 0:
            return x
        x += 1
        print('GRAPH BREAK', flush=True)
        return f2(x, n - 1)

    def f1(x: Tensor):
        n = 3  # assert n >= 0
        a = f2(x, n)
        return a + x

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break_at_first_statement():
    """
    Feature: test graph break in call_function.
    Description: graph break at first statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3():
        print('GRAPH BREAK', flush=True)  # break
        a = Tensor([2, 4, 6])
        return a + 1

    def f2():
        a = f3()
        return ops.mul(a, 2)

    def f1(x: Tensor, y: Tensor):
        z = f2()
        return x - y - z

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = f1(x, y)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = f1(x, y)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_is_dict_and_is_alive_local_v1():
    """
    Feature: test graph break in call_function.
    Description: param is dict, and it is alive local.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x: Tensor, d: dict):
        a = ops.add(d['a'], x)
        print('GRAPH BREAK', flush=True)  # break
        b = ops.sub(d['b'], x)
        return a, b

    def f1(x: Tensor, y: Tensor):
        d = {'a': x + y, 'b': x - y}
        a, b = f2(x, d)
        return a * b

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = f1(x, y)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = f1(x, y)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_is_dict_and_is_alive_local_v2():
    """
    Feature: test graph break in call_function.
    Description: param is dict, and it is alive local.
    Expectation: The result of PIJit is same as pynative.
    """

    d = {'a': Tensor([1., 2., 3.]), 'b': ops.randn(3)}

    def f2(x: Tensor):
        d2 = d
        a = ops.add(d2['a'], x)
        print('GRAPH BREAK', flush=True)  # break
        b = ops.sub(d2['b'], x)
        return a, b

    def f1(x: Tensor):
        a, b = f2(x)
        return a * b

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v1():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, y: int) -> Tensor:
        z = x * y
        z = z + x
        z = z / 2
        return z

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        c = c - a
        f = f3
        print('GRAPH BREAK', flush=True)
        return f

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f = f2(z, {'value': y[1]})
        result = f(z, y[0])
        return result

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v2():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """
    xxx = 0

    def f3(x: Tensor, y: int) -> Tensor:
        nonlocal xxx  # unsupported syntax, cannot convert f3 to AnfNode
        xxx += 1
        z = x * y
        z = z + x
        return z

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        return f3

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f = f2(z, {'value': y[1]})
        result = f(z, y[0])
        return result

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v3():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, y: int) -> Tensor:
        z = x * y
        print('GRAPH BREAK', end='\n\n')
        z = z + x
        return z

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        return f3, c

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f, c = f2(z, {'value': y[1]})
        print('GRAPH BREAK', end='\n\n')
        result = f(z, y[0])
        return result + c

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v4():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """
    VAR = 0.25

    def f3(x: Tensor, y: int) -> Tensor:
        z = x * y
        a = VAR
        print('GRAPH BREAK', end='\n\n')
        z = z + x
        return z * a

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        return f3, c

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f, c = f2(z, {'value': y[1]})
        result = f(z, y[0])
        return result + c

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


GLOBAL_SCALE = 2.0
GLOBAL_TENSOR = ops.ones((4, 4))


def f2(input_tensor: Tensor,
       weights: Tensor,
       params: dict,
       sizes: tuple,
       count: int,
       tensors_list: list) -> tuple:
    outer_tensor = input_tensor * weights

    def f3(tensor1: Tensor,
           tensor2: Tensor,
           scale: float,
           config: dict,
           dims: tuple,
           values: list,
           flag: bool) -> Tensor:
        scaled_tensor = tensor1 * GLOBAL_SCALE - count  # global var + free var
        print('GRAPH BREAK', end='')
        ret = ops.matmul(scaled_tensor, tensor2) + outer_tensor  # free var
        ret = ret + config['bias']

        if len(dims) > 2:
            ret = ret.reshape(dims)
            print('GRAPH BREAK', end='')
            ret += 1

        for val in values:
            ret = ret + ops.mean(val)
        return ret * scale

    new_config = {
        'bias': params['bias'] * 2,
        'scale': params['scale'] * 1.5
    }

    new_dims = sizes + (1,)
    if count > 5:
        scale_factor = 2.0
    else:
        scale_factor = 1.0

    print('GRAPH BREAK', end='')
    result = f3(input_tensor,
                weights,
                scale_factor,
                new_config,
                new_dims,
                tensors_list,
                True)

    return result, new_config, new_dims


def f1(x: Tensor,
       y: Tensor,
       batch_size: int,
       shape: tuple,
       config: dict,
       tensor_array: list) -> Tensor:
    intermediate = x - ops.add(y, GLOBAL_TENSOR)
    print('GRAPH BREAK', end='')
    new_params = {
        'bias': 0.5,
        'scale': config['learning_rate']
    }

    result, updated_config, dims = f2(intermediate,
                                      y,
                                      new_params,
                                      shape,
                                      batch_size,
                                      tensor_array)
    print('GRAPH BREAK', end='')
    return result * updated_config['scale']


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_many_params_and_many_alive_locals_and_free_vars():
    """
    Feature: test graph break in call_function.
    Description: param is dict, and it is alive local.
    Expectation: The result of PIJit is same as pynative.
    """

    x = ops.rand(4, 4)
    y = ops.rand(4, 4)
    batch_size = 8
    shape = (2, 8)
    config = {'learning_rate': 0.001}
    tensor_array = [ops.rand(2, 2) for _ in range(3)]

    o1 = f1(x, y, batch_size, shape, config, tensor_array)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y, batch_size, shape, config, tensor_array)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


class SetattrNetV1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.scalar = 5
        self.tensor = Tensor([1.0, 2.0, 3.0])
        self.tuple_attr = (10, 20)
        self.list_attr = [Tensor([2.0, 4.0, 6.0]), Tensor([1.0, 3.0, 5.0])]

    def construct(self, x):
        return self._layer1(x)

    def _layer1(self, x):
        self.scalar += 1
        self.tensor -= x
        result = self._layer2(x)
        self.scalar += 2
        return result + self.tensor

    def _layer2(self, x):
        self.list_attr = [self.list_attr[0] * 2, self.list_attr[1] / 2]
        result = self._layer3(x, self.tuple_attr[0])
        self.tuple_attr = (self.tuple_attr[0] - self.tuple_attr[1],)
        return result + self.list_attr[0]

    def _layer3(self, x, val):
        temp = self.tensor + val
        result = temp + self._layer4(x, val)
        self.tensor = temp - x
        return result

    def _layer4(self, x, val):
        temp = self.scalar * x - val
        print('GRAPH BREAK', end='\n\n')
        self.scalar -= 1
        return temp


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setattr_side_effect_v1():
    """
    Feature: test graph break in call_function.
    Description: Has setattr side-effect operations.
    Expectation: The result of PIJit is same as pynative.
    """

    x = Tensor([0.5, 1.0, 1.5])
    net1 = SetattrNetV1()
    o1 = net1(x)

    net2 = SetattrNetV1()
    net2.construct = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = net2(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(net2.construct, break_count=1)
    assert_equal(net1.scalar, net2.scalar)
    match_array(net1.tensor, net2.tensor, error=7)
    assert_equal(net1.list_attr, net2.list_attr)
    assert_equal(net1.tuple_attr, net2.tuple_attr)
