# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def compile_net(net, x, y, b):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


# model_parallel test
def test_sum_mul():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul1 = P.Mul()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.mul2 = P.Mul()

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, (0,))
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([32, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_sum_mul2():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul1 = P.Mul()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.mul2 = P.Mul()

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, (0, 1))
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")

    x = Tensor(np.ones([128, 128, 64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128, 64, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_sum_mul3():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul1 = P.Mul()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.mul2 = P.Mul()

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, -1)
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_net(net, x, y, b)
