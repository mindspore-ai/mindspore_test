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
from tests.mark_utils import arg_mark

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from .test_grad_of_dynamic import TestDynamicGrad


class BiasAddNet(nn.Cell):
    def __init__(self):
        super(BiasAddNet, self).__init__()
        self.bias_add = P.BiasAdd()

    def construct(self, x, y):
        return self.bias_add(x, y)


def run_dynamic_shape():
    test_dynamic = TestDynamicGrad(BiasAddNet())
    input_x = Tensor(np.arange(6).reshape((2, 3)), ms.float32)
    bias = Tensor(np.random.random(3).reshape((3,)), ms.float32)
    test_dynamic.test_dynamic_grad_net([input_x, bias])


def run_dynamic_rank():
    test_dynamic = TestDynamicGrad(BiasAddNet())
    input_x = Tensor(np.arange(6).reshape((2, 3)), ms.float32)
    bias = Tensor(np.random.random(3).reshape((3,)), ms.float32)
    test_dynamic.test_dynamic_grad_net([input_x, bias], True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_bias_add():
    """
    Feature: BiasAdd Grad DynamicShape.
    Description: Test case of dynamic shape for  BiasAdd grad operator.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_level="O0")
    run_dynamic_shape()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    run_dynamic_shape()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_bias_add():
    """
    Feature: BiasAdd Grad DynamicRank.
    Description: Test case of dynamic rank for  BiasAdd grad operator.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_level="O0")
    run_dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    run_dynamic_rank()
