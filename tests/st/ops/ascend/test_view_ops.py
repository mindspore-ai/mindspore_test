# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.context as context
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
import mindspore.ops as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self, tanspose_a=False, transpose_b=False):
        super(Net, self).__init__()
        self.transpose = P.TransposeView()
        self.matmul = P.MatMul(tanspose_a, transpose_b)

    def construct(self, x, perm, mat):
        out = self.transpose(x, perm)
        out = self.matmul(out, mat)
        return out


class NetSplit(nn.Cell):
    def __init__(self):
        super(NetSplit, self).__init__()
        self.split = P.Split(0, 2)
        self.matmul = P.MatMul()

    def construct(self, x):
        a, b = self.split(x)
        out = self.matmul(a, b)
        return out


class NetCat(nn.Cell):
    def __init__(self):
        super(NetCat, self).__init__()
        self.cat = P.Concat(axis=0)
        self.matmul = P.MatMul()

    def construct(self, x, y, z):
        a = self.matmul(x, y)
        b = self.matmul(x, z)
        c = self.cat((a, b))
        out = c + c
        out = out / 2.0
        return out

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_transpose_view():
    """
    Feature: Transpose view operation
    Description: test the Transpose kernel, with view operation.
    Expectation: the output is same with numpy
    """
    x = np.random.rand(1280, 256).astype(np.float32) / 10
    mat = np.random.rand(1280, 3840).astype(np.float32) / 10
    perm = (1, 0)

    net = Net()
    out = net(Tensor(x), perm, Tensor(mat))
    out_np = np.matmul(x.T, mat)
    assert np.allclose(out.asnumpy(), out_np, rtol=10e-4, atol=10e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_split_view():
    """
    Feature: Transpose view operation
    Description: test the Transpose kernel, with view operation.
    Expectation: the output is same with numpy
    """
    x = np.random.rand(256, 128).astype(np.float32) / 10

    net = NetSplit()
    out = net(Tensor(x))

    a, b = np.split(x, 2, 0)
    out_np = np.matmul(a, b)
    assert np.allclose(out.asnumpy(), out_np, rtol=10e-4, atol=10e-4)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_concat_view():
    """
    Feature: Concat view operation
    Description: test the Concat kernel, with view operation.
    Expectation: the output is same with numpy
    """
    x = np.random.rand(16, 16).astype(np.float32) / 10
    y = np.random.rand(16, 16).astype(np.float32) / 10
    z = np.random.rand(16, 16).astype(np.float32) / 10

    net = NetCat()
    out = net(Tensor(x), Tensor(y), Tensor(z))
    out_np = np.concatenate((np.matmul(x, y), np.matmul(x, z)))
    assert np.allclose(out.asnumpy(), out_np, rtol=10e-4, atol=10e-4)


class ViewOut(nn.Cell):
    '''net with view out'''
    def __init__(self):
        super(ViewOut, self).__init__()
        self.transpose = P.TransposeView()

    @ms.jit
    def construct(self, x):
        x = self.transpose(x, (0, 1, 2, 4, 3))
        res = ms.mint.select(x, 1, 2)
        return res


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_view_out():
    """
    Feature: Runtime view graph mode.
    Description: view op as graph output.
    Expectation: the output is same as pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.random.randn(2, 3, 4, 5, 6).astype(np.float32))
    net = ViewOut()
    out_graph = net(x)
    x = x.transpose((0, 1, 2, 4, 3))
    out_pynative = ms.mint.select(x, 1, 2)
    assert np.allclose(out_graph.asnumpy(), out_pynative.asnumpy(), rtol=10e-4, atol=10e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pynative_view_to_graph():
    """
    Feature: Runtime view graph mode.
    Description: view input from pynative.
    Expectation: the output is same as pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.random.randn(2, 3, 4, 5, 6).astype(np.float32))
    x = x.transpose((0, 1, 2, 4, 3))
    net = ViewOut()
    out_graph = net(x)
    x = x.transpose((0, 1, 2, 4, 3))
    out_pynative = ms.mint.select(x, 1, 2)
    assert np.allclose(out_graph.asnumpy(), out_pynative.asnumpy(), rtol=10e-4, atol=10e-4)


class MakeContiguous(nn.Cell):
    '''net with view to aclop'''
    def __init__(self):
        super(MakeContiguous, self).__init__()
        self.transpose = P.TransposeView()

    @ms.jit
    def construct(self, x):
        x = self.transpose(x, (0, 1, 2, 4, 3))
        res = ms.mint.select(x, 1, 2)
        res = res[:0] # StridedSlice is aclop
        return res


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_view_to_aclop():
    """
    Feature: Runtime view graph mode.
    Description: view input from pynative.
    Expectation: the output is same as pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.random.randn(2, 3, 4, 5, 6).astype(np.float32))
    net = MakeContiguous()
    out_graph = net(x)

    x = x.transpose((0, 1, 2, 4, 3))
    out_pynative = ms.mint.select(x, 1, 2)
    out_pynative = out_pynative[:0]
    assert np.allclose(out_graph.asnumpy(), out_pynative.asnumpy(), rtol=10e-4, atol=10e-4)
