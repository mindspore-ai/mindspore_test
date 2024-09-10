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
import mindspore.ops as P
from mindspore import Tensor, context, Parameter
from mindspore import nn
from mindspore import Layout
from mindspore.communication.management import init, get_rank
from mindspore.nn import Momentum, TrainOneStepCell
from mindspore.train.serialization import save_checkpoint, load_checkpoint

context.set_context(mode=context.GRAPH_MODE)
init()


def standalone_run(standalone_net, *inputs):
    standalone_out = standalone_net(*inputs)
    print("standalone_out is ", standalone_out.asnumpy())
    save_checkpoint(standalone_net, f"./standalone_{get_rank()}.ckpt")
    param_dict = load_checkpoint(f"./standalone_{get_rank()}.ckpt")
    return param_dict


def parallel_run(parallel_net, *inputs):
    parallel_out = parallel_net(*inputs)
    print("parallel_out is ", parallel_out.asnumpy())
    save_checkpoint(parallel_net, f"./parallel_{get_rank()}.ckpt")
    param_dict = load_checkpoint(f"./parallel_{get_rank()}.ckpt")
    return param_dict


def compare_params(standalone_params, parallel_params):
    for key in standalone_params.keys():
        assert np.allclose(standalone_params[key].asnumpy(), parallel_params[key].asnumpy(), atol=1e-3, rtol=1e-3)


def test_custom_op_all_tensor_no_tuple_param():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob_tuple_param(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul_tuple_param(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.custom_weight1 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight1")
            self.custom_weight2 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight2")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul_tuple_param, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob_tuple_param)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, self.custom_weight1, self.custom_weight2)
            out = self.relu(out)
            return out

    np.random.seed(5)
    x = Tensor(np.random.rand(8, 1).astype(np.float32))

    # stand alone
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    standalone_net = Net()
    optimizer_standalone = Momentum(standalone_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    standalone_net = TrainOneStepCell(standalone_net, optimizer_standalone)
    standalone_net.set_train()
    standalone_params = standalone_run(standalone_net, x)

    # parallel
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch", parallel_mode="semi_auto_parallel")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"), layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout=in_layout, out_layout=out_layout)
    optimizer_parallel = Momentum(parallel_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    parallel_net = TrainOneStepCell(parallel_net, optimizer_parallel)
    parallel_net.set_train()
    parallel_params = parallel_run(parallel_net, x)
    compare_params(standalone_params, parallel_params)


def test_custom_op_all_tensor_no_tuple():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, y, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, y, z)
            out = self.relu(out)
            return out

    np.random.seed(5)
    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    y = Tensor(np.random.rand(8, 8).astype(np.float32))
    z = Tensor(np.random.rand(8, 8).astype(np.float32))

    # stand alone
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    standalone_net = Net()
    optimizer_standalone = Momentum(standalone_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    standalone_net = TrainOneStepCell(standalone_net, optimizer_standalone)
    standalone_net.set_train()
    standalone_params = standalone_run(standalone_net, x, y, z)

    # parallel
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch", parallel_mode="semi_auto_parallel")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"), layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout=in_layout, out_layout=out_layout)
    optimizer_parallel = Momentum(parallel_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    parallel_net = TrainOneStepCell(parallel_net, optimizer_parallel)
    parallel_net.set_train()
    parallel_params = parallel_run(parallel_net, x, y, z)
    compare_params(standalone_params, parallel_params)


def test_custom_op_one_scalar():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, 1, z)
            out = self.relu(out)
            return out

    np.random.seed(5)
    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    z = Tensor(np.random.rand(8, 8).astype(np.float32))

    # stand alone
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    standalone_net = Net()
    optimizer_standalone = Momentum(standalone_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    standalone_net = TrainOneStepCell(standalone_net, optimizer_standalone)
    standalone_net.set_train()
    standalone_params = standalone_run(standalone_net, x, z)

    # parallel
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch", parallel_mode="semi_auto_parallel")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (layout("dp", "None"), layout("None"), layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout=in_layout, out_layout=out_layout)
    optimizer_parallel = Momentum(parallel_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    parallel_net = TrainOneStepCell(parallel_net, optimizer_parallel)
    parallel_net.set_train()
    parallel_params = parallel_run(parallel_net, x, z)
    compare_params(standalone_params, parallel_params)


def test_custom_op_two_scalar():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, 1, 2)
            out = self.relu(out)
            return out

    np.random.seed(5)
    x = Tensor(np.random.rand(8, 1).astype(np.float32))

    # stand alone
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    standalone_net = Net()
    optimizer_standalone = Momentum(standalone_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    standalone_net = TrainOneStepCell(standalone_net, optimizer_standalone)
    standalone_net.set_train()
    standalone_params = standalone_run(standalone_net, x)

    # parallel
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch", parallel_mode="semi_auto_parallel")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (layout("dp", "None"), layout("None"), layout("None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout=in_layout, out_layout=out_layout)
    optimizer_parallel = Momentum(parallel_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    parallel_net = TrainOneStepCell(parallel_net, optimizer_parallel)
    parallel_net.set_train()
    parallel_params = parallel_run(parallel_net, x)
    compare_params(standalone_params, parallel_params)
