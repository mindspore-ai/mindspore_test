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
# ============================================================================
""" test_custom_cpp_function """

import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, nn
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_custom_cpp_function_multi_input():
    """
    Feature: Custom cpp autograd function.
    Description: Custom forward function of multi input single output.
    Expectation: success.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = Parameter(2.0, requires_grad=True)
            self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_launch_aclnn_ops.cpp'],
                                          backend="Ascend").load()

        def construct(self, x, y):
            z = self.my_ops.mul(x, y)
            return self.my_ops.mul(z, self.p)

    x = Tensor(1.0, ms.float32) * 2
    y = Tensor(1.0, ms.float32) * 3
    net = MyNet()
    grad_op = ms.value_and_grad(net, grad_position=(0, 1), weights=net.trainable_params())
    out, grads = grad_op(x, y)
    assert np.allclose(out.asnumpy(), np.array([12.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[0][0].asnumpy(), np.array([6.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[0][1].asnumpy(), np.array([4.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([6.0], dtype=np.float32), 0.00001, 0.00001)
