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
# ============================================================================

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore.common.parameter import Parameter, ParameterTuple


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


def teardown_module():
    context.set_context(save_graphs=False)


class NetInner(nn.Cell):
    def __init__(self):
        super(NetInner, self).__init__()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()
        self.para1 = Parameter(Tensor([2, 3, 4, 5], ms.float32), name="para1")
        self.para2 = Parameter(Tensor([2, 3, 4, 5], ms.float32), name="para2")

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.addn((x, self.para1))
        x = self.relu(x)
        x = self.addn((x, self.para2))
        return x


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_parameter_requires_grad_changed():
    """
    Feature: PyNative auto dynamic check.
    Description: Check parameter requires grad.
    Expectation: The result is dynamic.
    """
    net = NetInner()
    grad_op = ops.GradOperation(get_all=False, get_by_list=True, sens_param=False)
    net_params = ParameterTuple(net.trainable_params())

    # run first time
    input_x = Tensor([1, 2, 3, 4], ms.float32) * 2
    input_y = Tensor([1, 2, 3, 4], ms.float32) * 3
    grad1 = grad_op(net, net_params)(input_x, input_y)
    assert len(grad1) == 2
    assert np.allclose(grad1[0].asnumpy(), Tensor(np.array([1, 1, 1, 1])).astype(np.float32).asnumpy(),
                       0.001, 0.001)
    assert np.allclose(grad1[1].asnumpy(), Tensor(np.array([1, 1, 1, 1])).astype(np.float32).asnumpy(),
                       0.001, 0.001)
    # run second time
    for p in net_params:
        p.requires_grad = False
        break
    net_params = ParameterTuple(net.trainable_params())
    grad2 = grad_op(net, net_params)(input_x, input_y)
    assert len(grad2) == 1
    assert np.allclose(grad2[0].asnumpy(), Tensor(np.array([1, 1, 1, 1])).astype(np.float32).asnumpy(),
                       0.001, 0.001)
