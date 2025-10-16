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
# pylint: disable=unused-variable
import os
import pytest
import subprocess
import numpy as np
import mindspore as ms


def generate_dyn(file_name, func_name, dyn_file_name, expected_num):
    if os.path.exists(dyn_file_name):
        os.remove(dyn_file_name)
    assert not os.path.exists(dyn_file_name)

    dirname = os.path.dirname(os.path.abspath(__file__))
    cmd = f"VLOG_v=1 python " + dirname + "/" + file_name + " " + func_name + " > " + dyn_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(dyn_file_name)
    with open(dyn_file_name, "r") as v_file:
        data = v_file.read()

    assert data.count("Start compiling") == expected_num
    assert data.count("End compiling") == expected_num
    os.remove(dyn_file_name)


def test_keyword_arguments():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn1", "dynamic_shape_fn1.log", 1)


def test_varargs_kwargs():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn2", "dynamic_shape_fn2.log", 1)


def test_decorator_callable():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn3", "dynamic_shape_fn3.log", 1)


def test_list_arguments():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn4", "dynamic_shape_fn4.log", 1)


def test_tuple_arguments():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn5", "dynamic_shape_fn5.log", 1)


def test_nested_tuple_arguments():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn6", "dynamic_shape_fn6.log", 1)


def test_with_dynamic():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic and dynamic=1.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn7", "dynamic_shape_fn7.log", 3)


def test_with_dynamic_and_tuple_inputs():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic and dynamic=1.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn8", "dynamic_shape_fn8.log", 3)


def test_invalid_usage():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(ValueError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(x=ms.Tensor(shape=[None, None], dtype=ms.float32))
        class Net(ms.nn.Cell):
            def construct(self, x):
                return x + 1
    assert "can only be used for function or method" in str(raise_info.value)


def test_invalid_args_static():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(TypeError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(x=ms.Tensor(np.ones([3, 4], np.float32)))
        def func(x):
            return x * x
    assert "When using decorator enable_dynamic, the shape of argument 'x' at least have one None" \
           in str(raise_info.value)


def test_invalid_args_undefined():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(ValueError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(y=ms.Tensor(shape=[4, None], dtype=ms.float32))
        def func(x):
            return x * x
    assert "'y' is not in list" in str(raise_info.value)


def test_invalid_args_number():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(ValueError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(x=ms.Tensor(shape=None, dtype=ms.int32), y=ms.Tensor(shape=None, dtype=ms.int32))
        def func(x):
            return x + 1
        func(ms.Tensor([1], ms.int32))
    assert "exceeds the number of function arguments" in str(raise_info.value)


def test_invalid_args_type():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(TypeError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(a=ms.Tensor(shape=None, dtype=ms.int32), b=ms.Tensor(shape=None, dtype=ms.int32))
        def func(a, b):
            return a + b
        func(ms.Tensor([1], ms.int32), 1)
    assert "the corresponding inputs only supports Tensor or" in str(raise_info.value)


def test_invalid_args_shape():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(ValueError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(a=ms.Tensor(shape=[2, None], dtype=ms.float32), b=ms.Tensor(shape=None, dtype=ms.float32))
        def func(a, b):
            return a + b
        a = ms.Tensor(np.random.randn(3, 3), ms.float32)
        b = ms.Tensor(np.random.randn(3, 3), ms.float32)
        func(a, b)
    assert "tensor shapes are not the same" in str(raise_info.value)


def test_invalid_args_dtype():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(TypeError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(a=ms.Tensor(shape=None, dtype=ms.int32), b=ms.Tensor(shape=None, dtype=ms.int32))
        def func(a, b):
            return a + b
        a = ms.Tensor(np.random.randn(1, 2), ms.int32)
        b = ms.Tensor(np.random.randn(3, 4), ms.float32)
        func(a, b)
    assert "tensor dtypes are not the same" in str(raise_info.value)


def test_invalid_args_int():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(TypeError) as raise_info:
        @ms.jit
        @ms.enable_dynamic(x=ms.mutable([ms.Tensor(shape=[None, 4], dtype=ms.float32), 2]))
        def func(x):
            return x[0] * x[1]
    assert "but the argument : x is type of:<class 'int'>." in str(raise_info.value)


def test_invalid_args_mutable():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    with pytest.raises(ValueError) as raise_info:
        d1 = ms.Tensor(shape=[None, 4], dtype=ms.float32)
        d2 = ms.Tensor(shape=[3, None], dtype=ms.float32)
        @ms.jit
        @ms.enable_dynamic(x=(d1, d2))
        def func(x):
            return x[0] * 2, x[1] * 3
        a = ms.Tensor(np.random.randn(1, 4), ms.float32)
        b = ms.Tensor(np.random.randn(3, 5), ms.float32)
        func((a, b))
    assert "should be mutable(tuple/list)" in str(raise_info.value)


def test_invalid_cell_in_graph_mode():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.num = 2

        @ms.jit
        @ms.enable_dynamic(x=ms.Tensor(shape=None, dtype=ms.int32))
        def construct(self, x, y):
            z = x + y
            return self.num * z

    ms.set_context(mode=ms.GRAPH_MODE)
    a = ms.Tensor(np.random.randn(1, 2), ms.int32)
    b = ms.Tensor(np.random.randn(1, 2), ms.int32)
    with pytest.raises(ValueError) as raise_info:
        net = Net()
        net(a, b)
    assert "the 'enable_dynamic' cannot be set" in str(raise_info.value)


def test_invalid_set_inputs():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    d = ms.Tensor(shape=None, dtype=ms.float32)
    class Net(ms.nn.Cell):
        @ms.jit
        @ms.enable_dynamic(x=d, y=d)
        def construct(self, x, y):
            out = x + x
            return out * y

    ms.set_context(mode=ms.PYNATIVE_MODE)
    a = ms.Tensor(np.random.randn(3, 4), ms.float32)
    b = ms.Tensor(np.random.randn(3, 4), ms.float32)
    net = Net()
    net.set_inputs(d, d)
    with pytest.raises(ValueError) as raise_info:
        net(a, b)
    assert "When `enable_dynamic` is provided, the `set_inputs()` cannot be set!" in str(raise_info.value)
