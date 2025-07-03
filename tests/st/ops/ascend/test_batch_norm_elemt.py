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
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore import mint


class BatchNormElemtNet(nn.Cell):
    def __init__(self):
        super(BatchNormElemtNet, self).__init__()
        self.batch_norm_elemt = mint.batch_norm_elemt

    def construct(self, input_data, weight, bias, mean, invstd, eps):
        return self.batch_norm_elemt(input_data, weight, bias, mean, invstd, eps)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_batch_norm_elemt_fwd(mode):
    """
    Feature: Test functional bns operator.
    Description: test bns.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    batch_norm_elemt_net = BatchNormElemtNet()

    input_data = Tensor(np.array([[1.], [2.], [3.]]), mstype.float32)
    weight = Parameter(Tensor(np.array([1.]), mstype.float32), name="weight")
    bias = Parameter(Tensor(np.array([10.]), mstype.float32), name="bias")

    mean = Tensor(np.array([2.]), mstype.float32)
    invstd = Tensor(np.array([2.]), mstype.float32)
    eps = 1e-5
    output = batch_norm_elemt_net(input_data, weight, bias, mean, invstd, eps)
    expected_output = np.array([[8.], [10.], [12.]], np.float32)
    assert np.allclose(output.numpy(), expected_output, rtol=0.005, atol=0.005)
