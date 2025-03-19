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


import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import auto_generate as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


a = Tensor(np.array([[[3, 0], [2, 1], [1, 0], [1, 1]],
                     [[3, 0], [2, 1], [1, 0], [1, 1]]], np.float32))
b = Tensor(np.array([[3, 1, 3, 4], [3, 1, 3, 4]], np.float32))


def compile_net(net):
    net.set_train()
    net(a, b)

    context.reset_auto_parallel_context()


class Net(Cell):
    def __init__(self, driver, strategy=None):
        super(Net, self).__init__()
        self.LstsqV2 = P.LstsqV2().shard(strategy)
        self.driver = driver

    def construct(self, x, y):
        return self.LstsqV2(x, y, self.driver)


def test_lstsqv2_auto_parallel():
    """
    Feature: test lstsqv2 auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=2,
                                      global_rank=0)
    net = Net(driver="gelsd")
    compile_net(net)


def test_lstsqv2_model_parallel():
    """
    Feature: test lstsqv2 model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    net = Net(driver="gelsd", strategy=((2, 1, 1), (2, 1)))
    compile_net(net)


def test_lstsqv2_strategy_error():
    """
    Feature: test invalid strategy for lstsqv2
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    net = Net(driver="gelsd", strategy=((2, 2, 1), (2, 1)))
    with pytest.raises(RuntimeError):
        compile_net(net)
