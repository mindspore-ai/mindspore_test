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
from mindspore import Parameter
from mindspore.common import dtype as mstype
from mindspore import mint


class BatchNormGatherStatsWithCountsNet(nn.Cell):
    def __init__(self):
        super(BatchNormGatherStatsWithCountsNet, self).__init__()
        self.batch_norm_gather_stats_with_counts = mint.batch_norm_gather_stats_with_counts

    def construct(self, input_data, mean, invstd, running_mean, running_var, momentum, eps,
                  counts):
        return self.batch_norm_gather_stats_with_counts(input_data, mean, invstd, running_mean, running_var, momentum,
                                                        eps, counts)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_batch_norm_gather_stats_with_counts_fwd(mode):
    """
    Feature: Test functional bns operator.
    Description: test bns.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    batch_norm_gather_stats_with_counts_net = BatchNormGatherStatsWithCountsNet()
    input_data = Tensor(np.linspace(0, 10, 2 * 3 * 12 * 12).reshape(2, 3, 12, 12), mstype.float32)
    mean = Tensor(np.linspace(0.1, 1, 4 * 3).reshape(4, 3), mstype.float32)
    invstd = Tensor(np.linspace(0.1, 1, 4 * 3).reshape(4, 3), mstype.float32)
    running_mean = Parameter(
        Tensor(np.linspace(0.1, 1, 3), mstype.float32), name="running_mean")
    running_var = Parameter(
        Tensor(np.linspace(0.1, 1, 3), mstype.float32), name="running_var")
    momentum = 1e-3
    eps = 1e-5
    counts = Tensor(np.array([4, 5, 6, 4]), mstype.float32)
    mean_all, invstd_all = batch_norm_gather_stats_with_counts_net(input_data, mean, invstd, running_mean,
                                                                   running_var, momentum, eps, counts)
    expected_mean_all = np.array([0.4746, 0.5565, 0.6383], np.float32)
    expected_invstd_all = np.array([0.2019, 0.3367, 0.4529], np.float32)
    expected_running_mean = np.array(
        [0.10037464, 0.5500065, 0.9996383], np.float32)
    expectec_running_var = np.array(
        [0.12579158, 0.5587633, 1.0041461], np.float32)
    assert np.allclose(mean_all.numpy(), expected_mean_all,
                       rtol=0.005, atol=0.005)
    assert np.allclose(invstd_all.numpy(),
                       expected_invstd_all, rtol=0.005, atol=0.005)
    assert np.allclose(running_mean.numpy(),
                       expected_running_mean, rtol=0.005, atol=0.005)
    assert np.allclose(running_var.numpy(),
                       expectec_running_var, rtol=0.005, atol=0.005)
