# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
from mindspore import ops
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self, bins, min_val, max_val):
        super(Net, self).__init__()
        self.histogram = ops.Histogram(bins=bins, min=min_val, max=max_val)

    def construct(self, x):
        return self.histogram(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_histogram_normal(mode):
    """
    Feature: Histogram
    Description: Verify the result of Histogram
    Expectation: success
    """
    context.set_context(mode=mode)
    bins, min_val, max_val = 4, 0.0, 3.0
    net = Net(bins, min_val, max_val)
    x = Tensor([1, 2, 1], mstype.int32)
    x2 = Tensor([1., 2., 1.], mstype.float32)
    x3 = Tensor([1., 2., 1.], mstype.float16)
    output = net(x)
    output2 = net(x2)
    output3 = net(x3)
    expected_output = np.array([0, 2, 1, 0])
    expected_output2 = np.array([0., 2., 1., 0.])
    assert np.array_equal(output.asnumpy(), expected_output)
    assert np.array_equal(output2.asnumpy(), expected_output2)
    assert np.array_equal(output3.asnumpy(), expected_output2)
