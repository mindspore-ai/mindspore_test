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

import numpy as np

from mindspore.common import dtype as mstype
from mindspore import Tensor, Parameter, context
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.ops as ops

from parallel.utils.utils import compile_net, ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        return self.grad_op(self.network)(*inputs)


class GradWrapTwoOutput(Cell):
    def __init__(self, network):
        super(GradWrapTwoOutput, self).__init__()
        self.network = network
        self.grad_op = C.GradOperation(get_all=True, sens_param=True)
        self.scale_sens = Parameter(1.0, name='scale_sens')

    def construct(self, *inputs):
        loss1, loss2 = self.network(*inputs)
        loss = loss1 + loss2
        scaling_sens_filled = C.ones_like(loss) * F.cast(self.scale_sens, F.dtype(loss))
        grads = self.grad_op(self.network)(*inputs, (scaling_sens_filled, scaling_sens_filled))
        return grads
        # return self.grad_op(self.network)(*inputs)


def test_multi_loss():
    """
    Feature: test multi loss with correctly insert VirtualDiv
    Description: The network has two loss output.
    Expectation: compile success and insert VirtualDiv correctly
    """
    class Net(Cell):
        def __init__(self, strategy=None):
            super(Net, self).__init__()
            self.relu = ops.ReLU().shard(strategy)
            self.reduce_sum = ops.ReduceSum(keep_dims=True).shard(strategy)

        def construct(self, x):
            out1 = self.relu(x)
            out1 = self.reduce_sum(out1) * 0.5
            out2 = self.reduce_sum(x) * 0.5
            return out1, out2

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    x = Tensor(np.random.normal(size=[32,]).astype(np.float32))
    strategy = ((8,),)
    net = GradWrapTwoOutput(Net(strategy))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Muls-0', ['_VirtualDiv-0'])
    assert validator.check_node_inputs_has('Muls-1', ['_VirtualDiv-1'])


def test_multi_loss_with_constant():
    """
    Feature: test multi loss with correctly insert VirtualDiv
    Description: The network has two loss output.
    Expectation: compile success and insert VirtualDiv correctly
    """
    class Net(Cell):
        def __init__(self, strategy=None):
            super(Net, self).__init__()
            self.relu = ops.ReLU().shard(strategy)
            self.reduce_sum = ops.ReduceSum(keep_dims=True).shard(strategy)
            self.fake_loss = Tensor([1.0], mstype.float32)

        def construct(self, x):
            out1 = self.relu(x)
            out1 = self.reduce_sum(out1) * 0.5
            return out1, self.fake_loss * 0.5

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    x = Tensor(np.random.normal(size=[32,]).astype(np.float32))
    strategy = ((8,),)
    net = GradWrapTwoOutput(Net(strategy))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Muls-0', ['_VirtualDiv-0'])


def test_multi_loss_with_depend():
    """
    Feature: test multi loss with correctly insert VirtualDiv
    Description: The network has two loss output.
    Expectation: compile success and insert VirtualDiv correctly
    """
    class Net(Cell):
        def __init__(self, strategy=None):
            super(Net, self).__init__()
            self.relu = ops.ReLU().shard(strategy)
            self.reduce_sum = ops.ReduceSum(keep_dims=True).shard(strategy)

        def construct(self, x):
            out1 = self.relu(x)
            out1 = self.reduce_sum(out1) * 0.5
            out2 = self.reduce_sum(x) * 0.5
            out2 = F.depend(out2, out1)
            return out1, out2

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    x = Tensor(np.random.normal(size=[32,]).astype(np.float32))
    strategy = ((8,),)
    net = GradWrapTwoOutput(Net(strategy))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Muls-0', ['_VirtualDiv-0'])
    assert validator.check_node_inputs_has('Muls-1', ['_VirtualDiv-1'])


def test_single_loss():
    """
    Feature: test single loss with correctly insert VirtualDiv
    Description: The network has one loss.
    Expectation: compile success and insert VirtualDiv correctly
    """
    class Net(Cell):
        def __init__(self, strategy=None):
            super(Net, self).__init__()
            self.relu = ops.ReLU().shard(strategy)
            self.reduce_sum = ops.ReduceSum(keep_dims=True).shard(strategy)

        def construct(self, x):
            out1 = self.relu(x)
            out1 = self.reduce_sum(out1)
            return out1

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    x = Tensor(np.random.normal(size=[32,]).astype(np.float32))
    strategy = ((8,),)
    net = GradWrap(Net(strategy))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('ReduceSum-0', ['_VirtualDiv-0'])


def test_single_loss_with_depend():
    """
    Feature: test single loss with correctly insert VirtualDiv
    Description: The network has one loss output with depend.
    Expectation: compile success and insert VirtualDiv correctly
    """
    class Net(Cell):
        def __init__(self, strategy=None):
            super(Net, self).__init__()
            self.relu = ops.ReLU().shard(strategy)
            self.gelu = ops.GeLU().shard(strategy)
            self.reduce_sum = ops.ReduceSum(keep_dims=True).shard(strategy)

        def construct(self, x):
            out1 = self.relu(x)
            tmp = self.gelu(x)
            out1 = self.reduce_sum(out1)
            out1 = F.depend(out1, tmp)
            return out1

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    x = Tensor(np.random.normal(size=[32,]).astype(np.float32))
    strategy = ((8,),)
    net = GradWrap(Net(strategy))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('ReduceSum-0', ['_VirtualDiv-0'])
