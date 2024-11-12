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
from mindspore import ops
from mindspore.mint.nn.functional import conv2d
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


class Net2d(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mint_conv2d = conv2d

    def construct(self, input_x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return self.mint_conv2d(input_x, weight, bias, stride, padding, dilation, groups)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_conv2d_default(mode):
    """
    Feature: mint.nn.functional.conv2d
    Description: Verify the result of conv2d
    Expectation: success
    Note: There is a precision problem on Ascend, #I6PT9L
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float32)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float32)
    weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                     [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float32)
    net = Net2d()
    output = net(x, weight, bias=bias)
    expected = np.array([[[[0.3286, -0.8062, -1.9410],
                           [-3.0758, -4.2105, -5.3453],
                           [-6.4801, -7.6149, -8.7497]],
                          [[2.7076, 2.9766, 3.2456],
                           [3.5145, 3.7835, 4.0525],
                           [4.3215, 4.5904, 4.8594]]],
                         [[[-20.0976, -21.2324, -22.3672],
                           [-23.5020, -24.6368, -25.7715],
                           [-26.9063, -28.0411, -29.1759]],
                          [[7.5492, 7.8182, 8.0871],
                           [8.3561, 8.6251, 8.8941],
                           [9.1630, 9.4320, 9.7010]]],
                         [[[-40.5238, -41.6586, -42.7934],
                           [-43.9282, -45.0630, -46.1978],
                           [-47.3326, -48.4673, -49.6021]],
                          [[12.3907, 12.6597, 12.9287],
                           [13.1977, 13.4666, 13.7356],
                           [14.0046, 14.2736, 14.5425]]]])
    assert np.allclose(output.asnumpy(), expected, atol=1e-4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_conv2d_padding_same(mode):
    """
    Feature: mint.nn.functional.conv2d
    Description: Verify the result of conv2d
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float32)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float32)
    weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                     [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float32)
    net = Net2d()
    output = net(x, weight, bias=bias, padding="same")
    expected = np.array([[[[0.3286, -0.8062, -1.9410],
                           [-3.0758, -4.2105, -5.3453],
                           [-6.4801, -7.6149, -8.7497]],
                          [[2.7076, 2.9766, 3.2456],
                           [3.5145, 3.7835, 4.0525],
                           [4.3215, 4.5904, 4.8594]]],
                         [[[-20.0976, -21.2324, -22.3672],
                           [-23.5020, -24.6368, -25.7715],
                           [-26.9063, -28.0411, -29.1759]],
                          [[7.5492, 7.8182, 8.0871],
                           [8.3561, 8.6251, 8.8941],
                           [9.1630, 9.4320, 9.7010]]],
                         [[[-40.5238, -41.6586, -42.7934],
                           [-43.9282, -45.0630, -46.1978],
                           [-47.3326, -48.4673, -49.6021]],
                          [[12.3907, 12.6597, 12.9287],
                           [13.1977, 13.4666, 13.7356],
                           [14.0046, 14.2736, 14.5425]]]])
    assert np.allclose(output.asnumpy(), expected, atol=1e-4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_conv2d_with_bf16():
    """
    Feature: The weight init of conv2d with type of bfloat16.
    Description: The weight init of conv 2d is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    x = ms.Tensor(np.ones([2, 2, 4, 4]), ms.bfloat16)
    weight = ms.Tensor(np.ones([2, 2, 1, 1]), ms.bfloat16)
    bias = ms.Tensor(np.ones([2]), ms.bfloat16)
    net = Net2d()
    output = net(x, weight, bias=bias)
    expected = np.array([[[3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.]],
                         [[3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.]]])
    cpu_cast = ops.Cast().set_device("CPU")
    output = cpu_cast(output, ms.float32)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_conv2d_dynamic():
    """
    Feature: mint.nn.functional.conv2d
    Description: dynamic shape and rank
    Expectation: success
    """
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    x1 = ms.Tensor(np.ones([2, 2, 4, 4]), ms.float16)
    weight1 = ms.Tensor(np.ones([2, 2, 1, 1]), ms.float16)
    x2 = ms.Tensor(np.ones([1, 2, 6, 8]), ms.float16)
    weight2 = ms.Tensor(np.ones([2, 2, 2, 3]), ms.float16)
    bias = ms.Tensor(np.ones([2]), ms.bfloat16)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    TEST_OP(conv2d, [[x1, weight1, bias, stride, padding, dilation, groups],
                     [x2, weight2, bias, stride, padding, dilation, groups]],
            'conv2d', disable_input_check=True, disable_yaml_check=True, disable_mode=['GRAPH_MODE'],
            disable_nontensor_dynamic_type="STATIC_LEN")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv2d_backward(context_mode):
    """
    Feature: mint.nn.functional.conv2d.
    Description: test conv2d op backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    net = Net2d()
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float32)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float32)
    weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                     [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float32)

    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias, stride, padding, dilation, groups)
    expected_x_grad = np.array([[[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]],
                                [[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]],
                                [[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]]])
    expected_weight_grad = np.array([[[[594.]], [[837.]]], [[[594.]], [[837.]]]])
    expected_bias_grad = np.array([27., 27.])
    assert np.allclose(grad_output[0].asnumpy(), expected_x_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[1].asnumpy(), expected_weight_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[2].asnumpy(), expected_bias_grad, atol=1e-4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv2d_vmap(context_mode):
    """
    Feature: mint.nn.functional.conv2d.
    Description: test conv2d op vmap.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=context_mode)
    ms.context.set_context(ascend_config={"conv_allow_hf32": False})
    x = Tensor(np.ones((3, 2, 2, 4, 4)), ms.float32)
    bias = Tensor(np.ones((2,)), ms.float32)
    weight = Tensor(np.ones((2, 2, 2, 2)), ms.float32)
    stride = 1
    padding = "same"
    dilation = 1
    groups = 1
    net = Net2d()
    in_axes = (0, None, None, None, None, None, None)
    net_vmap = ops.vmap(net, in_axes=in_axes, out_axes=0)
    out = net_vmap(x, weight, bias, stride, padding, dilation, groups)
    assert out.asnumpy().shape == (3, 2, 2, 4, 4)
