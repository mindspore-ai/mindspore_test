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

import numpy as np
from mindspore import context, nn, Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark

class AssignSubNet(nn.Cell):
    def __init__(self, ref, ref_reduce):
        super().__init__()
        self.unique = P.Unique()
        self.assign_sub = P.AssignSub()
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.ref = Parameter(ref)
        self.cast = P.Cast()
        self.ref_reduce = Parameter(ref_reduce)

    def construct(self, x, indices):
        x = self.assign_sub(self.ref, x)
        unique_indices, _ = self.unique(indices)
        dtype = x.dtype
        x = self.cast(x, mstype.float64)
        x = self.reducesum(x, unique_indices)
        x = self.cast(x, dtype)
        out = self.assign_sub(self.ref_reduce, x)
        return x, out


class AssignSubDynamicShape():
    def __init__(self, input_ref_np, input_value_np, indices_np):
        self.input_ref_np = input_ref_np
        self.input_value_np = input_value_np
        self.indices_np = indices_np
        self.unique = P.Unique()
        self.ref_reduce_np = None
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.cast = P.Cast()

    def impl(self):
        indices_ms = Tensor(self.indices_np)
        unique_indices, _ = self.unique(indices_ms)
        ref_reduce = Tensor(self.input_ref_np)
        dtype = ref_reduce.dtype
        ref_reduce = self.cast(ref_reduce, mstype.float32)
        ref_reduce = self.reducesum(ref_reduce, unique_indices.asnumpy().tolist())
        ref_reduce = self.cast(ref_reduce, dtype)
        self.ref_reduce_np = ref_reduce.asnumpy()
        net = AssignSubNet(Tensor(self.input_ref_np), ref_reduce)
        input_ms = Tensor(self.input_value_np)
        input_dyn = Tensor(shape=[None for _ in input_ms.shape], dtype=input_ms.dtype)
        net.set_inputs(input_dyn, indices_ms)
        out_ms = net(input_ms, indices_ms)
        return out_ms


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_single_op_graph():
    """
    Feature: Runtime special output.
    Description: Test the output is the depend with value node, that the value can't be converted the tensor.
    Expectation: Not throw exception.
    """
    np.random.seed(0)
    input_ref_np = np.random.randn(128, 128, 32).astype(np.float16)
    input_value_np = np.random.randn(128, 128, 32).astype(np.float16)
    indices_np = np.random.randint(0, 2, size=2)
    context.set_context(device_target="CPU")
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = AssignSubDynamicShape(input_ref_np, input_value_np, indices_np)
    result1 = net1.impl()

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    net2 = AssignSubDynamicShape(input_ref_np, input_value_np, indices_np)
    result2 = net2.impl()
    assert np.allclose(result1[0].asnumpy(), result2[0].asnumpy())
