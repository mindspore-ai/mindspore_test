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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import mint


class Net3d(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mint_conv3d = mint.nn.functional.conv3d

    def construct(self, input_x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return self.mint_conv3d(input_x, weight, bias, stride, padding, dilation, groups)


@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_conv3d_default(mode):
    """
    Feature: mint.nn.functional.conv3d
    Description: Verify the result of conv3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})

    x = Tensor(np.linspace(0, 5, 1 * 2 * 4 * 4 * 4),
               ms.float32).reshape(1, 2, 4, 4, 4)
    bias = Tensor([0.1, 0.2], ms.float32)
    weight = Tensor(np.linspace(0, 5, 2 * 2 * 2 * 2 * 2),
                    ms.float32).reshape(2, 2, 2, 2, 2)
    net = Net3d()
    output = net(x, weight, bias=bias)
    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias)

    expect_output = [[[[[47.3441, 48.1061, 48.8681], [50.3921, 51.1541, 51.9161], [53.4401, 54.2021, 54.9641]],
                       [[59.5361, 60.2981, 61.0601], [62.5841, 63.3461, 64.1081], [65.6321, 66.3941, 67.1561]],
                       [[71.7281, 72.4901, 73.2521], [74.7761, 75.5382, 76.3002], [77.8242, 78.5862, 79.3482]]],
                      [[[116.5322, 118.9198, 121.3074], [126.0827, 128.4703, 130.8579], [135.6331, 138.0207, 140.4083]],
                       [[154.7339, 157.1215, 159.5091], [164.2843, 166.6719, 169.0595], [173.8347, 176.2224, 178.6100]],
                       [[192.9356, 195.3232, 197.7108], [202.4860, 204.8736, 207.2612],
                        [212.0364, 214.4240, 216.8116]]]]]

    expect_input_grad = [[[[[2.5806, 5.4839, 5.4839, 2.9032], [5.8065, 12.2581, 12.2581, 6.4516],
                            [5.8065, 12.2581, 12.2581, 6.4516], [3.2258, 6.7742, 6.7742, 3.5484]],
                           [[6.4516, 13.5484, 13.5484, 7.0968], [14.1935, 29.6774, 29.6774, 15.4839],
                            [14.1935, 29.6774, 29.6774, 15.4839], [7.7419, 16.1290, 16.1290, 8.3871]],
                           [[6.4516, 13.5484, 13.5484, 7.0968], [14.1935, 29.6774, 29.6774, 15.4839],
                            [14.1935, 29.6774, 29.6774, 15.4839], [7.7419, 16.1290, 16.1290, 8.3871]],
                           [[3.8710, 8.0645, 8.0645, 4.1935], [8.3871, 17.4194, 17.4194, 9.0323],
                            [8.3871, 17.4194, 17.4194, 9.0323], [4.5161, 9.3548, 9.3548, 4.8387]]],
                          [[[5.1613, 10.6452, 10.6452, 5.4839], [10.9677, 22.5806, 22.5806, 11.6129],
                            [10.9677, 22.5806, 22.5806, 11.6129], [5.8065, 11.9355, 11.9355, 6.1290]],
                           [[11.6129, 23.8710, 23.8710, 12.2581], [24.5161, 50.3226, 50.3226, 25.8065],
                            [24.5161, 50.3226, 50.3226, 25.8065], [12.9032, 26.4516, 26.4516, 13.5484]],
                           [[11.6129, 23.8710, 23.8710, 12.2581], [24.5161, 50.3226, 50.3226, 25.8065],
                            [24.5161, 50.3226, 50.3226, 25.8065], [12.9032, 26.4516, 26.4516, 13.5484]],
                           [[6.4516, 13.2258, 13.2258, 6.7742], [13.5484, 27.7419, 27.7419, 14.1935],
                            [13.5484, 27.7419, 27.7419, 14.1935], [7.0968, 14.5161, 14.5161, 7.4194]]]]]
    expect_bias_grad = [27., 27.]
    expect_weight_grad = [[[[[22.3228, 23.3858], [26.5748, 27.6378]],
                            [[39.3307, 40.3937], [43.5827, 44.6457]]],
                           [[[90.3543, 91.4173], [94.6063, 95.6693]],
                            [[107.3622, 108.4252], [111.6142, 112.6772]]]],
                          [[[[22.3228, 23.3858], [26.5748, 27.6378]],
                            [[39.3307, 40.3937], [43.5827, 44.6457]]],
                           [[[90.3543, 91.4173], [94.6063, 95.6693]],
                            [[107.3622, 108.4252], [111.6142, 112.6772]]]]]

    assert np.allclose(output.asnumpy(), expect_output, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[0].asnumpy(), expect_input_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[1].asnumpy(), expect_weight_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[2].asnumpy(), expect_bias_grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.skip(reason="Has not supported.")
def test_ops_conv3d_batchfy(mode):
    """
    Feature: mint.nn.functional.conv3d
    Description: Verify the result of conv3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})

    x = Tensor(np.linspace(0, 5, 2 * 4 * 4 * 4),
               ms.float32).reshape(2, 4, 4, 4)
    bias = Tensor([0.1, 0.2], ms.float32)
    weight = Tensor(np.linspace(0, 5, 2 * 2 * 2 * 2 * 2),
                    ms.float32).reshape(2, 2, 2, 2, 2)
    net = Net3d()
    output = net(x, weight, bias=bias)
    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias)

    expect_output = [[[[47.3441, 48.1061, 48.8681], [50.3921, 51.1541, 51.9161], [53.4401, 54.2021, 54.9641]],
                      [[59.5361, 60.2981, 61.0601], [62.5841, 63.3461, 64.1081], [65.6321, 66.3941, 67.1561]],
                      [[71.7281, 72.4901, 73.2521], [74.7761, 75.5382, 76.3002], [77.8242, 78.5862, 79.3482]]],
                     [[[116.5322, 118.9198, 121.3074], [126.0827, 128.4703, 130.8579], [135.6331, 138.0207, 140.4083]],
                      [[154.7339, 157.1215, 159.5091], [164.2843, 166.6719, 169.0595], [173.8347, 176.2224, 178.6100]],
                      [[192.9356, 195.3232, 197.7108], [202.4860, 204.8736, 207.2612],
                       [212.0364, 214.4240, 216.8116]]]]

    expect_input_grad = [[[[[2.5806, 5.4839, 5.4839, 2.9032], [5.8065, 12.2581, 12.2581, 6.4516],
                            [5.8065, 12.2581, 12.2581, 6.4516], [3.2258, 6.7742, 6.7742, 3.5484]],
                           [[6.4516, 13.5484, 13.5484, 7.0968], [14.1935, 29.6774, 29.6774, 15.4839],
                            [14.1935, 29.6774, 29.6774, 15.4839], [7.7419, 16.1290, 16.1290, 8.3871]],
                           [[6.4516, 13.5484, 13.5484, 7.0968], [14.1935, 29.6774, 29.6774, 15.4839],
                            [14.1935, 29.6774, 29.6774, 15.4839], [7.7419, 16.1290, 16.1290, 8.3871]],
                           [[3.8710, 8.0645, 8.0645, 4.1935], [8.3871, 17.4194, 17.4194, 9.0323],
                            [8.3871, 17.4194, 17.4194, 9.0323], [4.5161, 9.3548, 9.3548, 4.8387]]],
                          [[[5.1613, 10.6452, 10.6452, 5.4839], [10.9677, 22.5806, 22.5806, 11.6129],
                            [10.9677, 22.5806, 22.5806, 11.6129], [5.8065, 11.9355, 11.9355, 6.1290]],
                           [[11.6129, 23.8710, 23.8710, 12.2581], [24.5161, 50.3226, 50.3226, 25.8065],
                            [24.5161, 50.3226, 50.3226, 25.8065], [12.9032, 26.4516, 26.4516, 13.5484]],
                           [[11.6129, 23.8710, 23.8710, 12.2581], [24.5161, 50.3226, 50.3226, 25.8065],
                            [24.5161, 50.3226, 50.3226, 25.8065], [12.9032, 26.4516, 26.4516, 13.5484]],
                           [[6.4516, 13.2258, 13.2258, 6.7742], [13.5484, 27.7419, 27.7419, 14.1935],
                            [13.5484, 27.7419, 27.7419, 14.1935], [7.0968, 14.5161, 14.5161, 7.4194]]]]]
    expect_bias_grad = [27., 27.]
    expect_weight_grad = [[[[[22.3228, 23.3858], [26.5748, 27.6378]],
                            [[39.3307, 40.3937], [43.5827, 44.6457]]],
                           [[[90.3543, 91.4173], [94.6063, 95.6693]],
                            [[107.3622, 108.4252], [111.6142, 112.6772]]]],
                          [[[[22.3228, 23.3858], [26.5748, 27.6378]],
                            [[39.3307, 40.3937], [43.5827, 44.6457]]],
                           [[[90.3543, 91.4173], [94.6063, 95.6693]],
                            [[107.3622, 108.4252], [111.6142, 112.6772]]]]]

    assert np.allclose(output.asnumpy(), expect_output, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[0].asnumpy(), expect_input_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[1].asnumpy(), expect_weight_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[2].asnumpy(), expect_bias_grad, atol=1e-4, rtol=1e-4)
