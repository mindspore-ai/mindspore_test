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
import mindspore as ms
from mindspore import nn, Tensor, ops
import mindspore.context as context
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_value_depend_infer():
    """
    Feature: Dynamic type.
    Description: Value depend in any type.
    Expectation: Not throw exception.
    """
    from mindspore import numpy as np
    class ValueDependInferNet(nn.Cell):
        def __init__(self, tau=1, hard=False):
            super().__init__()
            self.tau = tau
            self.hard = hard
            self.zero = Tensor(0, ms.float32)
            self.one = Tensor(1, ms.float32)

        def construct(self, logits: Tensor):
            eps = 1e-10
            dim = -1
            U = ops.uniform(logits.shape, self.zero, self.one)
            noise = -ops.log(-ops.log(U + eps) + eps)
            y = logits + noise
            y_soft = ops.Softmax()(y / self.tau)
            if self.hard:
                index = y_soft.argmax(dim)
                y_hard = ops.OneHot()(index, 2, self.one, self.zero)
                ret = y_hard
            else:
                ret = y_soft
            return ret

    class EyeNet(nn.Cell):
        def __init__(self, dtype=ms.float32):
            super(EyeNet, self).__init__()
            self.dtype = dtype
            self.insert_graph = ValueDependInferNet()

        def construct(self, x):
            def update_shape(baize_out, ex_shape):
                if isinstance(baize_out, (list, tuple)):
                    baize_out = baize_out[0]
                if baize_out.shape == ex_shape:
                    return baize_out
                out_size = baize_out.numel()
                ex_size = 1
                for dim in ex_shape:
                    ex_size *= dim
                if out_size > ex_size:
                    return ops.flatten(baize_out, start_dim=0)[:ex_size].reshape(ex_shape)
                if out_size < ex_size:
                    return ops.pad(ops.flatten(baize_out, start_dim=0),
                                   Tensor([0, ex_size - out_size]), mode='constant',
                                   value=0.1).reshape(ex_shape)
                return baize_out.reshape(ex_shape)

            baize_out = ops.eye(x.shape[1], 3, dtype=self.dtype)
            logits = update_shape(baize_out, (64, 64))
            baize_out = self.insert_graph(logits)
            return baize_out

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    net = EyeNet()
    input_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_dyn)
    x = Tensor(np.ones([3, 3]), dtype=ms.float32)
    out = net(x).asnumpy()
    assert out.shape == (64, 64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_tuple_arg_to_dynamic_tuple_and_partial_para():
    """
    Feature: Dynamic type.
    Description: Value depend in any type.
    Expectation: Not throw exception.
    """
    import numpy as np
    class Net(nn.Cell):
        def construct(self, list_in):
            length = len(list_in)
            if length >= 2:
                ele1 = list_in[0]
                ele2 = list_in[length - 1]
                tmp = ops.add(ele1, ele2)
                return (ele1, ele2, tmp)
            add = ops.add(list_in[0], 1)
            return (list_in[0], add)

    input1 = np.random.rand(2, 2).astype(np.float32)
    input2 = np.random.rand(2, 2).astype(np.float32)
    inputx = ms.mutable((Tensor(input1), Tensor(input2)), dynamic_len=True)
    gradnet = ms.ops.GradOperation(get_all=True)(Net())
    _ = gradnet(inputx)
