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
import mindspore.context as context
from mindspore import Tensor, nn
import mindspore.ops.operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.slice = P.Slice()
        self.strided_slice = P.StridedSlice()
        self.abs = P.Abs()

    def construct(self, x, y):
        a = x + y
        b = x - y
        c = self.slice(a, (10, 0, 133), (4, 2, 233))
        d = self.strided_slice(b, (10, 0, 133), (14, 2, 233), (1, 1, 1))
        e = self.abs(d)
        f = c * c
        return e, f


def get_output(net, i0, i1, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(jit_level='O1')
        context.set_context(
            graph_kernel_flags="--enable_cluster_ops=Slice,StridedSlice")
    else:
        context.set_context(jit_level='O0')
    net_obj = net()
    output = net_obj(i0, i1)
    return output


def fuse(dtype):
    i0 = Tensor(np.random.uniform(1, 2, [37, 4, 1025]).astype(dtype))
    i1 = Tensor(np.random.uniform(1, 2, [37, 4, 1025]).astype(dtype))
    expect = get_output(Net, i0, i1, False)
    output = get_output(Net, i0, i1, True)
    if dtype == np.float32:
        eps = 1e-5
    elif dtype == np.float16:
        eps = 1e-3
    else:
        eps = 0
    assert np.allclose(expect[0].asnumpy(), output[0] .asnumpy(), eps, eps)
    assert np.allclose(expect[1].asnumpy(), output[1] .asnumpy(), eps, eps)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
def test_slice_dvm(dtype):
    """
    Feature: test slice case for graph_kernel in Ascend.
    Description: ascend test case, use graph_kernel execute ops.
    Expectation: the result match with close graph_kernel result
    """
    context.set_context(mode=context.GRAPH_MODE)
    fuse(dtype)
