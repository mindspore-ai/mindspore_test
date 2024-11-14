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
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.ops.auto_generate.gen_ops_prim import BatchNormElemtGrad
from tests.mark_utils import arg_mark

class BatchNormElemtGradNet(Cell):
    def __init__(self):
        super(BatchNormElemtGradNet, self).__init__()
        self.batch_norm_reduce_grad = BatchNormElemtGrad()

    def construct(self, dout, input_x, mean_param, invstd_param, weight, inputG, weightG, biasG):
        return self.batch_norm_reduce_grad(dout, input_x, mean_param, invstd_param, weight, inputG, weightG, biasG)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    batch_norm_reduce_grad_net = BatchNormElemtGradNet()

    input_x = Tensor([[[[0., 0.2173913],
                        [0.4347826, 0.65217394]],
                       [[0.8695652, 1.0869565],
                        [1.3043479, 1.5217391]],
                       [[1.7391304, 1.9565217],
                        [2.173913, 2.3913043]]],
                      [[[2.6086957, 2.826087],
                        [3.0434783, 3.2608695]],
                       [[3.4782608, 3.6956522],
                        [3.9130435, 4.130435]],
                       [[4.347826, 4.5652175],
                        [4.7826085, 5.]]]])
    dout = Tensor([[[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                   [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
    mean = Tensor([1.6304348, 2.5, 3.3695652])
    invstd = Tensor([0.7536912, 0.7536912, 0.7536912])
    weight = Tensor([2., 2., 2.])
    sum_dy = Tensor([8., 8., 8.])
    sum_dy_xmu = Tensor([0., 0., 0.])
    count_all_param = Tensor([[2.], [2.], [2.]])

    grad_input = batch_norm_reduce_grad_net(
        dout, input_x, mean, invstd, weight, sum_dy, sum_dy_xmu, count_all_param)
    expected_grad_input = np.array([[[[-0.5025, -0.5025], [-0.5025, -0.5025]],
                                     [[-0.5025, -0.5025], [-0.5025, -0.5025]],
                                     [[-0.5025, -0.5025], [-0.5025, -0.5025]]],
                                    [[[-0.5025, -0.5025], [-0.5025, -0.5025]],
                                     [[-0.5025, -0.5025], [-0.5025, -0.5025]],
                                     [[-0.5025, -0.5025], [-0.5025, -0.5025]]]], np.float32)
    assert np.allclose(grad_input.numpy(),
                       expected_grad_input, rtol=1e-4, atol=1e-4)
