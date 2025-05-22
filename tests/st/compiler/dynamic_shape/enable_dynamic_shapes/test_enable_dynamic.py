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
import os
import pytest
import subprocess
import numpy as np
import mindspore as ms
from tests.mark_utils import arg_mark


def generate_dyn(file_name, func_name, dyn_file_name):
    if os.path.exists(dyn_file_name):
        os.remove(dyn_file_name)
    assert not os.path.exists(dyn_file_name)

    cmd = f"VLOG_v=1 python " + file_name + " " + func_name + " > " + dyn_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(dyn_file_name)
    with open(dyn_file_name, "r") as v_file:
        data = v_file.read()

    assert data.count("Start compiling") == 1
    assert data.count("End compiling") == 1
    os.remove(dyn_file_name)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_keyword_arguments():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn1", "dynamic_shape_fn1.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_varargs_kwargs():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn2", "dynamic_shape_fn2.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_decorator_callable():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn3", "dynamic_shape_fn3.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_arguments():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_dynamic.py", "fn4", "dynamic_shape_fn4.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
