# Copyright 2022 Huawei Technologies Co., Ltd
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
import copy
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.ops import operations as P
from mindspore.nn.optim.adam import _update_run_op
from tests.mark_utils import arg_mark

context.set_context(jit_level='O0')


class OriNet(nn.Cell):
    """Origin net uses _update_run_op"""

    def __init__(self, decay_flag):
        super(OriNet, self).__init__()
        self.decay_flag = decay_flag
        self.optim_filter = True

    def construct(self, param, m, v, lr, beta1, beta2, eps, weight_decay, gradient):
        next_param = _update_run_op(beta1, beta2, eps, lr, weight_decay, param, m, v, gradient,
                                    self.decay_flag, self.optim_filter)
        return next_param


class FissionNet(nn.Cell):
    """Fission net uses P.AdamWeightDecay()"""

    def __init__(self):
        super(FissionNet, self).__init__()
        self.optim_filter = True

    def construct(self, param, m, v, lr, beta1, beta2, eps, weight_decay, gradient):
        if self.optim_filter:
            adam = P.AdamWeightDecay()
            next_param = adam(param, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
            return next_param
        return gradient


def test_adam_weight_decay_fission_1_decay_flag_is_true():
    """
    Feature: AdamWeightDecay op
    Description: test the rightness of AdamWeightDecay kernel, decay_flag is true
    Expectation: the output is wrong
    """
    decay_flag = True  # equivalent to weight_decay is not zero
    weight_decay = Parameter(Tensor(np.array([0.9]).astype(np.float32)), name="weight_decay")
    beta1 = Parameter(Tensor(np.array([0.9]).astype(np.float32)), name="beta1")
    beta2 = Parameter(Tensor(np.array([0.999]).astype(np.float32)), name="beta2")
    eps = Parameter(Tensor(np.array([1e-8]).astype(np.float32)), name="eps")
    lr = Parameter(Tensor(np.array([0.001]).astype(np.float32)), name="lr")
    gradient = Parameter(Tensor(np.array([[2, 3], [1, 5]]).astype(np.float32)), name="gradient")

    # The inputs: param, m and v will be modified in-place by P.AdamWeightDecay() or _update_run_op(),
    # so here defines two copied of them: (param1, m1, v1) and (param2, m2, v2)
    param1 = Parameter(Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32)), name="param1")
    m1 = Parameter(Tensor(np.array([[5, 6], [7, 8]]).astype(np.float32)), name="m1")
    v1 = Parameter(Tensor(np.array([[3, 1], [7, 4]]).astype(np.float32)), name="v1")

    param2 = copy.deepcopy(param1)
    m2 = copy.deepcopy(m1)
    v2 = copy.deepcopy(v1)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
    origin_net = OriNet(decay_flag)
    output1 = origin_net(param1, m1, v1, lr, beta1, beta2, eps, weight_decay, gradient)
    fission_net = FissionNet()
    output2 = fission_net(param2, m2, v2, lr, beta1, beta2, eps, weight_decay, gradient)
    assert (output1.asnumpy() == output2[0].asnumpy()).all()


def test_adam_weight_decay_fission_2_decay_flag_is_false():
    """
    Feature: AdamWeightDecay op
    Description: test the rightness of ScaleGrad kernel, decay_flag is false
    Expectation: the output is wrong
    """
    decay_flag = False  # equivalent to weight_decay is zero
    weight_decay = Parameter(Tensor(np.array([0]).astype(np.float32)), name="weight_decay")
    beta1 = Parameter(Tensor(np.array([0.9]).astype(np.float32)), name="beta1")
    beta2 = Parameter(Tensor(np.array([0.999]).astype(np.float32)), name="beta2")
    eps = Parameter(Tensor(np.array([1e-8]).astype(np.float32)), name="eps")
    lr = Parameter(Tensor(np.array([0.001]).astype(np.float32)), name="lr")
    gradient = Parameter(Tensor(np.array([[2, 3], [1, 5]]).astype(np.float32)), name="gradient")

    # The inputs: param, m and v will be modified in-place by P.AdamWeightDecay() or _update_run_op(),
    # so here defines two copied of them: (param1, m1, v1) and (param2, m2, v2)
    param1 = Parameter(Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32)), name="param1")
    m1 = Parameter(Tensor(np.array([[5, 6], [7, 8]]).astype(np.float32)), name="m1")
    v1 = Parameter(Tensor(np.array([[3, 1], [7, 4]]).astype(np.float32)), name="v1")

    param2 = copy.deepcopy(param1)
    m2 = copy.deepcopy(m1)
    v2 = copy.deepcopy(v1)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
    origin_net = OriNet(decay_flag)
    output1 = origin_net(param1, m1, v1, lr, beta1, beta2, eps, weight_decay, gradient)
    fission_net = FissionNet()
    output2 = fission_net(param2, m2, v2, lr, beta1, beta2, eps, weight_decay, gradient)
    assert (output1.asnumpy() == output2[0].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_adam_weight_decay_pass_without_same_type():
    """
    Feature: AdamWeightDecay op
    Description: test the rightness of AdamWeightDecay kernel, decay_flag is true
    Expectation: the output is same
    """
    decay_flag = True  # equivalent to weight_decay is not zero
    weight_decay = Parameter(Tensor(np.array([0.9]).astype(np.float32)), name="weight_decay")
    beta1 = Parameter(Tensor(np.array([0.9]).astype(np.float32)), name="beta1")
    beta2 = Parameter(Tensor(np.array([0.999]).astype(np.float32)), name="beta2")
    eps = Parameter(Tensor(np.array([1e-8]).astype(np.float32)), name="eps")
    lr = Parameter(Tensor(np.array([0.001]).astype(np.float32)), name="lr")
    gradient = Parameter(Tensor(np.array([[2, 3], [1, 5]]).astype(np.float32)), name="gradient")

    # The inputs: param, m and v will be modified in-place by P.AdamWeightDecay() or _update_run_op(),
    # so here defines two copied of them: (param1, m1, v1) and (param2, m2, v2)
    param1 = Parameter(Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32)), name="param1")
    m1 = Parameter(Tensor(np.array([[5, 6], [7, 8]]).astype(np.float16)), name="m1")
    v1 = Parameter(Tensor(np.array([[3, 1], [7, 4]]).astype(np.float16)), name="v1")

    param2 = copy.deepcopy(param1)
    m2 = copy.deepcopy(m1)
    v2 = copy.deepcopy(v1)

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    origin_net = OriNet(decay_flag)
    output1 = origin_net(param1, m1, v1, lr, beta1, beta2, eps, weight_decay, gradient)
    fission_net = FissionNet()
    output2 = fission_net(param2, m2, v2, lr, beta1, beta2, eps, weight_decay, gradient)
    assert (output1.asnumpy() == output2[0].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_adam_weight_decay_pass_with_same_type_to_assign():
    """
    Feature: AdamWeightDecay op
    Description: test the rightness of AdamWeightDecay kernel, decay_flag is true
    Expectation: the output is same
    """
    decay_flag = True  # equivalent to weight_decay is not zero
    weight_decay = Parameter(Tensor(np.array([0.9]).astype(np.float32)), name="weight_decay")
    beta1 = Parameter(Tensor(np.array([0.9]).astype(np.float32)), name="beta1")
    beta2 = Parameter(Tensor(np.array([0.999]).astype(np.float32)), name="beta2")
    eps = Parameter(Tensor(np.array([1e-8]).astype(np.float32)), name="eps")
    lr = Parameter(Tensor(np.array([0.001]).astype(np.float32)), name="lr")
    gradient = Parameter(Tensor(np.array([[2, 3], [1, 5]]).astype(np.float32)), name="gradient")

    # The inputs: param, m and v will be modified in-place by P.AdamWeightDecay() or _update_run_op(),
    # so here defines two copied of them: (param1, m1, v1) and (param2, m2, v2)
    param1 = Parameter(Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32)), name="param1")
    m1 = Parameter(Tensor(np.array([[5, 6], [7, 8]]).astype(np.float32)), name="m1")
    v1 = Parameter(Tensor(np.array([[3, 1], [7, 4]]).astype(np.float32)), name="v1")

    param2 = copy.deepcopy(param1)
    m2 = copy.deepcopy(m1)
    v2 = copy.deepcopy(v1)

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    origin_net = OriNet(decay_flag)
    output1 = origin_net(param1, m1, v1, lr, beta1, beta2, eps, weight_decay, gradient)
    fission_net = FissionNet()
    output2 = fission_net(param2, m2, v2, lr, beta1, beta2, eps, weight_decay, gradient)
    assert (output1.asnumpy() == output2[0].asnumpy()).all()
