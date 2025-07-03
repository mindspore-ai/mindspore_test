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
from mindspore.ops.auto_generate import BroadcastToView, ExpandDimsView, NarrowView, SelectExtView, SplitTensorView
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

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_broadcast_to_view():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class BroadcastToViewNet(nn.Cell):
        def __init__(self):
            super(BroadcastToViewNet, self).__init__()
            self.broadcast_to_view = BroadcastToView()

        def construct(self, x):
            output = self.broadcast_to_view(x, (2, 3))
            return output

    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    net = BroadcastToViewNet()
    graph_output = net(x)

    pynative_output = BroadcastToView()(x, (2, 3))
    assert (graph_output.asnumpy() == pynative_output.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_expand_dims_view():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class ExpandDimsViewNet(nn.Cell):
        def __init__(self):
            super(ExpandDimsViewNet, self).__init__()
            self.expand_dims_view = ExpandDimsView()

        def construct(self, x):
            output = self.expand_dims_view(x, 0)
            return output

    x = Tensor(np.array([[2, 2], [2, 2]]), ms.float32)
    net = ExpandDimsViewNet()
    graph_output = net(x)

    pynative_output = ExpandDimsView()(x, 0)
    assert (graph_output.asnumpy() == pynative_output.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_narrow_view():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class NarrowViewNet(nn.Cell):
        def __init__(self):
            super(NarrowViewNet, self).__init__()
            self.narrow_view = NarrowView()

        def construct(self, x):
            output = self.narrow_view(x, 0, 0, 2)
            return output

    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ms.int32)
    net = NarrowViewNet()
    graph_output = net(x)

    pynative_output = NarrowView()(x, 0, 0, 2)
    assert (graph_output.asnumpy() == pynative_output.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_transpose_ext_view():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class TransposeExtViewNet(nn.Cell):
        def __init__(self):
            super(TransposeExtViewNet, self).__init__()
            self.transpose_ext_view = P.TransposeExtView()

        def construct(self, x):
            output = self.transpose_ext_view(x, 0, 2)
            return output

    context.set_context(jit_config={"jit_level": "O0"})
    x = Tensor(np.ones((2, 3, 4), dtype=np.float32))
    net = TransposeExtViewNet()
    output = net(x)
    assert output.shape == (4, 3, 2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_view_and_inplace_nested_ctrl_dynamic_rank():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class DynamicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reducesum = P.ReduceSum()
            self.expanddimsview = ExpandDimsView()
            self.selectview = SelectExtView()
            self.splittensorview = SplitTensorView()

        def construct(self, x, y):
            if self.reducesum(x) < 3 * self.reducesum(y):
                x.add_(y)
            else:
                y = self.expanddimsview(y, 1)
            if x.shape == (2, 4, 8):
                x = self.selectview(x, 0, 1)
                y = self.splittensorview(y, 2, 0)
            return x, y

    context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    with pytest.raises(ValueError) as raise_info:
        x_np = np.ones([2, 4, 8]).astype(np.int32)
        input_x = Tensor(x_np)
        y_np = 2 * np.ones([2, 4, 8]).astype(np.int32)
        input_y = Tensor(y_np)
        net = DynamicNet()
        out = net(input_x, input_y)
        print("out: ", out)
    assert "Unsupported dynamic shape for graph mode." in str(raise_info.value)
