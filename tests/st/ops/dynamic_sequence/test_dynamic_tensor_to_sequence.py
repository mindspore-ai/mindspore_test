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
import pytest

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops.operations import _sequence_ops as seq
from mindspore import context
from mindspore import Tensor
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.tensor_to_tuple = seq.TensorToTuple()
        self.tensor_to_scalar = seq.TensorToScalar()

    def construct(self, x, y):
        return self.tensor_to_tuple(x), self.tensor_to_scalar(y)


def dyn_case():
    x = Tensor([1, 2, 3], mstype.int64)
    y = Tensor(1, mstype.float32)
    expect_x = (1, 2, 3)
    expect_y = 1.0
    net = Net()
    res_x, res_y = net(x, y)
    assert expect_x == res_x
    assert expect_y == res_y


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_seq_to_tensor():
    """
    Feature: test TensorToxx.
    Description: inputs is dynamic sequence or scalar.
    Expectation: the result match with numpy result
    """
    dyn_case()
