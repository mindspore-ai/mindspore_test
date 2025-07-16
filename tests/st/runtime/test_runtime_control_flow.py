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
import mindspore
from mindspore import Tensor, ops, nn, context
import mindspore.ops.operations as P
import numpy as np
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_partition_graph_between_shape_and_reshape_1():
    """
    Feature: graph partition for control flow.
    Description: base scene.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y, z):
            shape = P.Shape()(x)
            shape_ele_1 = shape[0]
            shape_ele_2 = shape[1]
            if z < 5:
                y = y + 2
                shape_res = shape_ele_1 - shape_ele_2
            else:
                y = y - 1
                shape_res = shape_ele_1 + shape_ele_2
            res = y * 2
            return ops.reshape(res, (shape_ele_2, shape_ele_1)), shape_res

    x_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z = Tensor(2, mindspore.float32)
    net = Net()
    net.set_inputs(x_dyn, y_dyn, z)
    net.construct = mindspore.jit(net.construct, backend="ms_backend")
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    out = net(x, y, z)
    print(out)


def test_partition_graph_between_shape_and_reshape_2():
    """
    Feature: graph partition for control flow.
    Description: base scene.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y, z):
            shape = P.Shape()(x)
            shape_ele_1 = shape[0]
            shape_ele_2 = shape[1]
            new_shape = (shape_ele_2, 1, shape_ele_1)
            if z < 5:
                y = y + 2
                shape_res = shape_ele_1 - shape_ele_2
                res2 = ops.reshape(x, new_shape)
            else:
                y = y - 1
                shape_res = shape_ele_1 + shape_ele_2
                res2 = ops.reshape(y, new_shape)
            res = y * 2
            return ops.reshape(res, new_shape), shape_res, res2

    x_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z = Tensor(2, mindspore.float32)
    net = Net()
    net.set_inputs(x_dyn, y_dyn, z)
    net.construct = mindspore.jit(net.construct, backend="ms_backend")
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    out = net(x, y, z)
    print(out)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_control_flow_if_input_heter():
    """
    Feature: test control flow if input heter.
    Description: test input heter in control flow.
    Expectation: No exception.
    """
    class ConditionNet(nn.Cell):
        def construct(self, x, y):
            if x > 1:
                x = x + 1
            else:
                x = x + 2
            return x + y

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    x = np.array([5], np.float32)
    y = np.array([3], np.float32)
    net = ConditionNet()
    add = P.Add().add_prim_attr("primitive_target", "CPU")
    x = add(Tensor(x), 1)
    out = net(x, Tensor(y))
    print('output: ', out)
