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
"""Test enable_dynamic api"""
# pylint: disable=unused-variable
import os
import glob
import pytest
import shutil
import subprocess
import numpy as np
import mindspore as ms
from mindspore import nn, context
from tests.mark_utils import arg_mark


def generate_dyn(file_name, func_name, dyn_file_name):
    if os.path.exists(dyn_file_name):
        os.remove(dyn_file_name)
    assert not os.path.exists(dyn_file_name)

    dirname = os.path.dirname(os.path.abspath(__file__))
    cmd = "VLOG_v=1 python " + dirname + "/" + file_name + " " + func_name + " > " + dyn_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(dyn_file_name)
    with open(dyn_file_name, "r", encoding='utf-8') as v_file:
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_in_graph_mode():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic.
    Expectation: Raise expected exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.num = 2

        @ms.jit
        @ms.enable_dynamic(x=ms.Tensor(shape=None, dtype=ms.int32))
        def construct(self, x, y):
            z = x + y
            return self.num * z

    context.set_context(mode=context.GRAPH_MODE)

    a = ms.Tensor(np.random.randn(1, 2), ms.int32)
    b = ms.Tensor(np.random.randn(1, 2), ms.int32)
    with pytest.raises(ValueError) as raise_info:
        net = Net()
        net(a, b)
    assert "the 'enable_dynamic' cannot be set" in str(raise_info.value)


def save_ir(ir_path):
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "1"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path


def check_ir(expect_ir_num, ir_path, expect_dict):
    try:
        ir_files = sorted(glob.glob(os.path.join(ir_path, '*validate*.ir')))
        assert len(ir_files) == expect_ir_num
        if len(ir_files) == 2:
            file = ir_files[1]
        elif len(ir_files) == 1:
            file = ir_files[0]
        elif len(ir_files) == 3:
            file = ir_files[2]
        else:
            raise RuntimeError("Check ir failed.")
        for key in expect_dict:
            cmd = f"grep '^%para' {file} | grep '{key}' | wc -l"
            output = subprocess.check_output(cmd, shell=True)
            output = str(output, 'utf-8').strip()
            assert int(output) == expect_dict[key]

    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        if os.path.exists(ir_path):
            shutil.rmtree(ir_path)


def test_jit_enable_dynamic_002():
    """
    Feature: ms.enable_dynamic
    Description: Test ms.enable_dynamic with jit
    Expectation: No exception.
    """
    case_name = "test_jit_enable_dynamic_002"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    save_ir(ir_path)

    d = ms.Tensor(shape=[None, 4], dtype=ms.float32)

    @ms.jit(backend="ms_backend")
    @ms.enable_dynamic(x=d)
    def my_mul(x):
        return x * x

    def my_mul_nojit(x):
        return x * x

    tensors = [
        ms.Tensor(np.random.randn(2, 4), ms.float32),
        ms.Tensor(np.random.randn(3, 4), ms.float32),
        ms.Tensor(np.random.randn(4, 4), ms.float32)
    ]
    for tensor in tensors:
        my_mul(tensor)
    check_ir(1, ir_path, {"(-1, 4)": 1})
