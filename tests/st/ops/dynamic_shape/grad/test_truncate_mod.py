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


class TruncateModNet(nn.Cell):
    def __init__(self):
        super(TruncateModNet, self).__init__()
        self.truncate_mode = P.TruncateMod()

    def construct(self, x, y):
        return self.truncate_mode(x, y)


def run_dynamic_shape():
    test_dynamic = TestDynamicGrad(TruncateModNet())
    x = Tensor(np.array([2, 4, -1]), ms.int32)
    y = Tensor(np.array([3, 3, 3]), ms.int32)
    test_dynamic.test_dynamic_grad_net([x, y])


def run_dynamic_rank():
    test_dynamic = TestDynamicGrad(TruncateModNet())
    x = Tensor(np.array([2, 4, -1]), ms.int32)
    y = Tensor(np.array([3, 3, 3]), ms.int32)
    test_dynamic.test_dynamic_grad_net([x, y], True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dynamic_truncate_mode():
    """
    Feature: TruncateMod Grad DynamicShape.
    Description: Test case of dynamic shape for  TruncateMod grad operator.
    Expectation: success.
    """
    context.set_context(jit_level='O0')
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE)
    run_dynamic_shape()
    run_dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    run_dynamic_shape()
    run_dynamic_rank()
