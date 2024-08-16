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
from tests.mark_utils import arg_mark
from mindspore import Tensor, nn, ops, context
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mm = ops.MatMul(transpose_a=True)
        self.cast = ops.Cast()

    def construct(self, x, y, z):
        o = self.mm(x, y)
        return self.cast(o, mstype.float32) + z


def get_output(net, x, y, z, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(jit_config={"jit_level": "O1"})
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    else:
        context.set_context(jit_config={"jit_level": "O0"})
    net_obj = net()
    output = net_obj(x, y, z)
    return output


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("shape1, shape2", [((4096, 1024), (4096, 2048)), ((4000, 1000), (4000, 3000))])
def test_dvm_matmul_fusion(shape1, shape2):
    """
    Feature: test matmul case for graph_kernel in Ascend.
    Description: ascend test case, use graph_kernel execute ops.
    Expectation: the result match with close graph_kernel result
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.random.uniform(1, 2, shape1).astype(np.float16))
    y = Tensor(np.random.uniform(1, 2, shape2).astype(np.float16))
    z = Tensor(np.random.uniform(1, 2, [shape2[1]]).astype(np.float32))
    expect = get_output(Net, x, y, z, False)
    output = get_output(Net, x, y, z, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 2e-3, 2e-3)
