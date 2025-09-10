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
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore import Tensor, nn, ops, Parameter
from tests.mark_utils import arg_mark

def register_saved_tensors_hook(pack_fn, unpack_fn):
    def decorator(func):
        setattr(func, "pack_fn", pack_fn)
        setattr(func, "unpack_fn", unpack_fn)
        return func

    return decorator


def unpack_prefetch(count):
    def decorator(func):
        setattr(func, "count", count)
        return func

    return decorator


def pack(tensor):
    return tensor.to(device="CPU")


def unpack(tensor):
    return tensor.to(device="NPU")


@unpack_prefetch(5)
@register_saved_tensors_hook(pack, unpack)
def my_func(x, w):
    x = x * x + w * x
    x *= x
    return x


class TestOffloadRegister(Cell):
    def construct(self, x, w):
        out = my_func(x, w)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_offload_register():
    """
    Feature: Support offload.
    Description: Support offload.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="test_offload_register")
    input_x = ms.Tensor([1, 1])
    input_w = ms.Tensor([2, 2])
    net = TestOffloadRegister()
    out = net(input_x, input_w)
    print("out: ", out)

class GradOfFirstInput(nn.Cell):
    def __init__(self, net):
        super(GradOfFirstInput, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

class TestOffloadRegisterGrad(Cell):
    def construct(self, x, w):
        y = x * w
        w = 2 * x + 3
        out = my_func(y, w)
        return out

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_offload_register_grad():
    """
    Feature: Support offload.
    Description: Support offload.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=3, save_graphs_path="test_offload_register_grad")
    x = ms.Tensor([1], dtype=mstype.float32)
    y = ms.Tensor([2], dtype=mstype.float32)
    output = GradOfFirstInput(TestOffloadRegisterGrad())(x, y)
    print("output:", output)


class TestOffloadTensorTo(Cell):
    def __init__(self):
        super(TestOffloadTensorTo, self).__init__()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name="z")

    def construct(self, x):
        # self.z.data = self.z.to(device="Ascend")
        # out = x * self.z
        # return out
        return self.z.to(device="Ascend") * x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_to():
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="test_parameter_to")
    x = ms.Tensor([2], dtype=mstype.float32)
    net = TestOffloadTensorTo()
    out = net(x)
    print("out: ", out)
