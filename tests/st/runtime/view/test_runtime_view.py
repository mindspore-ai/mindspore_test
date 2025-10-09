# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, ops
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_output_view_parameter():
    """
    Feature: Support view.
    Description: Support view.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.atleast_2d = ops.atleast_2d

        def construct(self, *args):
            return self.atleast_2d(args)

    input_x = Tensor(np.random.randn(4, 4, 4), mstype.float64)
    net = Net()
    context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    res1 = net(*input_x)
    context.set_context(mode=ms.PYNATIVE_MODE)
    res2 = net(*input_x)
    assert np.allclose(res1[3].asnumpy(), res2[3].asnumpy())
