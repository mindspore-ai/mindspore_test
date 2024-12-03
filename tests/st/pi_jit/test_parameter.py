# coding=utf-8

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
""" test nn.Parameter """

from mindspore import context, jit, Tensor, ops, nn, mutable

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_no_graph_break, assert_equal

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_Parameter_in_nested_tuple_list_or_dict():
    """
    Feature: Parameter parsing.
    Description: Parameter in nested tuple list or dict.
    Expectation: result is right, no graph break.
    """

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(4, 4, has_bias=False)
            self.all_params = [{'params': self.trainable_params(), 'lr': 1e-5, 'weight_decay': 0.01}]

        def construct(self, x: Tensor, y: Tensor):
            sz = y.shape[0]
            return self.inner(x, sz, self.all_params)

        def inner(self, x: Tensor, sz: int, params):
            return ops.matmul(x * sz, params[0]['params'][0])

    model1 = Model()

    model2 = Model()
    model2.dense = model1.dense
    model2.all_params = model1.all_params
    model2.construct = jit(model2.construct, mode='PIJit', jit_config=jit_cfg)

    x = ops.rand(2, 4)
    y = ops.rand(3, 3)
    o1 = model1(x, y)
    o2 = model2(x, y)
    assert_equal(o1, o2)
    assert_no_graph_break(model2.construct, call_count=1)

    ops.assign_add(model2.dense.weight, ops.ones_like(model2.dense.weight))
    o1 = model1(x, y)
    o2 = model2(x, y)
    assert_equal(o1, o2)
    assert_no_graph_break(model2.construct, call_count=2)
