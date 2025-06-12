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
"""Test grad jit with filter"""
import os
import shutil
import subprocess
import numpy as np
from mindspore import Tensor, Parameter, ParameterTuple, jit, ops, context
from mindspore._extends.parse import compile_config
import mindspore.nn as nn
from tests.mark_utils import arg_mark


def get_from_ir_before_filter(para, save_graphs_path):
    output_before = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" %
         (para, os.path.join(save_graphs_path, "opt_backward_[0-9]*.ir"))],
        shell=True)
    out_before = str(output_before, 'utf-8').strip()
    return out_before


def get_from_ir_after_filter(para, save_graphs_path):
    output_after = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" %
         (para, os.path.join(save_graphs_path, "filtered_output_grad_fg_[0-9]*.ir"))],
        shell=True)
    out_after = str(output_after, 'utf-8').strip()
    return out_after

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_psjit():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    try:
        save_graphs_path = "./test_jit_grad_with_filter"
        os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = save_graphs_path
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([3])
        ret1 = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
        assert ret1 == 1
        ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
        assert ret2 == 1
        para = '= PrimFunc_Muls(%'
        out_before = get_from_ir_before_filter(para, save_graphs_path)
        assert out_before == "3"
        out_after = get_from_ir_after_filter(para, save_graphs_path)
        assert out_after == "0"
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        compile_config.GRAD_JIT_FILTER = ""
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_psjit_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret == 1
    ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
    assert ret2 == 1
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_psjit_with_multiple_position():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
    assert isinstance(ret1, tuple)
    assert len(ret1) == 2
    assert ret1[0] == 1
    assert ret1[1] == 2
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_psjit_with_multiple_position_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    try:
        save_graphs_path = "./test_jit_grad_with_filter"
        os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = save_graphs_path
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([3])
        ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
        assert isinstance(ret1, tuple)
        assert len(ret1) == 2
        assert ret1[0] == 1
        assert ret1[1] == 2
        ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
        assert isinstance(ret2, tuple)
        assert len(ret2) == 2
        assert ret2[0] == 1
        assert ret2[1] == 2
        para = '= PrimFunc_Muls(%'
        out_before = get_from_ir_before_filter(para, save_graphs_path)
        assert out_before == "3"
        out_after = get_from_ir_after_filter(para, save_graphs_path)
        assert out_after == "1"
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        compile_config.GRAD_JIT_FILTER = ""
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_psjit_with_different_position():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret1 == 1
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_pijit():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret1 == 1
    ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
    assert ret2 == 1
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_pijit_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret == 1
    ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
    assert ret2 == 1
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_pijit_with_multiple_position():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
    assert isinstance(ret1, tuple)
    assert len(ret1) == 2
    assert ret1[0] == 1
    assert ret1[1] == 2
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_pijit_with_multiple_position_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
    assert isinstance(ret1, tuple)
    assert len(ret1) == 2
    assert ret1[0] == 1
    assert ret1[1] == 2
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter_grad_jit_pijit_with_different_position():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret1 == 1
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


class ParamNet(nn.Cell):
    def __init__(self):
        super(ParamNet, self).__init__()
        self.w = Parameter(Tensor([2., 2.]), name="w")
        self.z = Parameter(Tensor([3., 3.]), name="z")
    @jit
    def construct(self, x):
        res = 2 * x + 3 * self.w + 4 * self.z
        return res

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_weights():
    """
    Features: GradOperation and grad.
    Description: Test F.grad with different weights twice in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 2]).astype(np.float32))
    try:
        save_graphs_path = "./test_jit_grad_with_filter"
        os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = save_graphs_path
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)
        net = ParamNet()
        weights1 = ParameterTuple(net.trainable_params()[:1])
        expect1 = np.array([2, 2]).astype(np.float32)
        out1 = ops.grad(net, weights=weights1)(x)  # pylint: disable=not-callable
        assert np.allclose(out1[0].asnumpy(), expect1)
        para = '= PrimFunc_Mul('
        out_before = get_from_ir_before_filter(para, save_graphs_path)
        assert out_before == "3"
        out_after = get_from_ir_after_filter(para, save_graphs_path)
        assert out_after == "2"
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        compile_config.GRAD_JIT_FILTER = ""
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)
