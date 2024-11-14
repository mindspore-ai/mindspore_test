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
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore import mint


class BatchNormStatsNet(nn.Cell):
    def __init__(self):
        super(BatchNormStatsNet, self).__init__()
        self.batch_norm_stats = mint.batch_norm_stats

    def construct(self, input_data, eps):
        return self.batch_norm_stats(input_data, eps)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_batch_norm_stats_fwd(mode):
    """
    Feature: Test functional bns operator.
    Description: test bns.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    batch_norm_stats_net = BatchNormStatsNet()

    input_data = Tensor(
        np.arange(3 * 2 * 3 * 4).reshape(3, 2, 3, 4), mstype.float32)
    eps = 1e-5
    output_mean, output_invstd = batch_norm_stats_net(input_data, eps)
    expected_mean = np.array([29.5000, 41.5000], np.float32)
    expected_invstd = np.array([0.0503, 0.0503], np.float32)

    assert np.allclose(output_mean.numpy(), expected_mean,
                       rtol=0.005, atol=0.005)
    assert np.allclose(output_invstd.numpy(),
                       expected_invstd, rtol=0.005, atol=0.005)

    input_data = Tensor(
        np.arange(3 * 2 * 3 * 4).reshape(3, 2, 3, 4), mstype.float16)
    eps = 1e-5
    output_mean, output_invstd = batch_norm_stats_net(input_data, eps)
    expected_mean = np.array([29.5000, 41.5000], np.float32)
    expected_invstd = np.array([0.0503, 0.0503], np.float32)

    assert np.allclose(output_mean.numpy(), expected_mean,
                       rtol=0.005, atol=0.005)
    assert np.allclose(output_invstd.numpy(),
                       expected_invstd, rtol=0.005, atol=0.005)
