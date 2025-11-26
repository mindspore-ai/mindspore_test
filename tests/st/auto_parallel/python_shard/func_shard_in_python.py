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

"""func shard in python"""
import time
import pytest
import numpy as np
import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
import mindspore.communication.management as D
import mindspore.common.dtype as mstype
from mindspore import nn, Tensor, Parameter
from mindspore.parallel import Layout
from mindspore.parallel import init_parameters
from mindspore.common.initializer import initializer
from mindspore import ops
from mindspore.parallel.spmd.shard import shard
from tests.st.auto_parallel.utils import create_dtensor, global_to_local, local_to_global


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


class ReLUNet(nn.Cell):
    """relu net"""
    def construct(self, x):
        x = ms.mint.nn.functional.relu(x)
        return x


class MatMulNet(nn.Cell):
    """Net composed of several ReLUs"""
    def construct(self, x, w):
        return ops.matmul(x, w)


class SimpleNet(nn.Cell):
    """Net composed of several ReLUs"""
    def __init__(self, strategy_list):
        super().__init__()
        self.cell_list = ms.nn.CellList()
        for strategy in strategy_list:
            relu_net = ms.mint.nn.ReLU()
            shard(relu_net, sharding_plan = strategy)
            self.cell_list.append(relu_net)

    def construct(self, x):
        for cell in self.cell_list:
            x = x.contiguous()
            x = cell(x)
        return x


class SimpleModel(nn.Cell):
    """simple model"""
    def __init__(self, input_size, output_size, strategy_list, w_layout=None, gamma_layout=None):
        super().__init__()
        self.weight = ms.Parameter(
            ms.parallel.DTensor.from_local(Tensor(np.random.randn(input_size, output_size).astype(np.float32)),
                                           w_layout),
            name='weight'
        )
        self.gamma = ms.Parameter(
            ms.parallel.DTensor.from_local(Tensor(np.random.randn(output_size).astype(np.float32)), gamma_layout),
            name='gamma'
        )
        self.cell_list = ms.nn.CellList()
        for strategy in strategy_list:
            relu_net = ms.mint.nn.ReLU()
            shard(relu_net, sharding_plan=strategy)
            self.cell_list.append(relu_net)

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x, _ = ms.ops.rms_norm(x, self.gamma)
        for cell in self.cell_list:
            x = cell(x)
        return x


def print_layout_info(tensor, name):
    """print_layout_info"""
    if hasattr(tensor, 'layout') and tensor.layout is not None:
        layout_dict = tensor.layout.to_dict()
        print(f"{name} Layout:")
        print(f"  device_matrix: {layout_dict['device_matrix']}")
        print(f"  tensor_map: {layout_dict['tensor_map']}")
        print(f"  alias_name: {layout_dict['alias_name']}")
        print(f"  rank_list: {layout_dict['rank_list'][:8]}...")  # 只显示前8个rank
    else:
        print(f"{name} has no layout information")


def run_scenario(scenario_name, x_layout, x_shape, strategy_list):
    """run_scenario"""
    D.init()
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print('='*80)

    # Create Data
    np_x = np.random.randn(*x_shape).astype(np.float32)

    # Create Dtensor
    x = create_dtensor(np_x, x_layout)
    print_layout_info(x, "Input X")

    # Create Net
    net = SimpleNet(strategy_list=strategy_list)
    output = net(x)
    print_layout_info(output, "Output")

    return output


def run_scenario_with_bprop(x_layout, w_layout, target_layout, strategy_list):
    """run_scenario_with_bprop"""
    D.init()
    input_size = 256
    output_size = 128
    batch_size = 4
    learning_rate = 0.01
    epochs = 10

    model = SimpleModel(input_size, output_size, strategy_list, w_layout = w_layout, gamma_layout = w_layout("None"))
    loss_fn = nn.MSELoss(reduction='none')

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = ms.parallel.parallelize_value_and_grad(forward_fn, optimizer.parameters)
    np_x = np.random.randn(batch_size, input_size).astype(np.float32)
    np_target = np.random.randn(batch_size, output_size).astype(np.float32)
    x = create_dtensor(np_x, x_layout)
    target = create_dtensor(np_target, target_layout)
    print_layout_info(x, "Input X")
    print_layout_info(model.weight, "Input w")
    print_layout_info(target, "Input target")
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x, target)
        with NoFallbackGuard():
            optimizer(grads)
        end = time.time()
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss_value}, Time: {end - start}")


base_device_matrix = (2, 4)  # dp=2, mp=4
base_alias_name = ("dp", "mp")
base_rank_list = list(range(8))

base_device_matrix2 = (8,)
base_alias_name2 = ("dp_mp",)
base_rank_list2 = list(range(8))

base_device_matrix3 = (2, 2, 2)
base_alias_name3 = ("cp", "ep", "tp")
base_rank_list3 = list(range(8))


def test_func_shard_1():
    '''
    Feature: func shard in python shard.
    Description: Test func shard in python shard with constant device_matrix.
    Expectation: Run success.
    '''
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = layout("dp", "None")

    in_strategy_1 = (layout("dp", "mp"),)
    out_strategy_1 = None

    in_strategy_2 = (layout("None", "None"),)
    out_strategy_2 = (layout("dp", "mp"),)

    in_strategy_3 = (layout("mp", "dp"),)
    out_strategy_3 = (layout("mp", "None"),)

    strategy_list = ({ "forward": { "input": in_strategy_1, "output": out_strategy_1}},
                     { "forward": { "input": in_strategy_2, "output": out_strategy_2}},
                     { "forward": { "input": in_strategy_3, "output": out_strategy_3}})
    output = run_scenario(
        "Data Parallel (DP)",
        x_layout,
        x_shape=(16, 256),
        strategy_list=strategy_list
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (0, -1)
    output_shape = output.shape
    assert output_shape == (32, 256)


def test_func_shard_2():
    '''
    Feature: Model parallel in python shard.
    Description: Test model parallel in python shard with changeable device matrix .
    Expectation: Run success.
    '''
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    layout2 = Layout(base_device_matrix2, base_alias_name2, base_rank_list2)
    x_layout = layout("dp", "mp")

    in_strategy_1 = (layout("None", "None"),)
    out_strategy_1 = None

    in_strategy_2 = (layout2("dp_mp", "None"),)
    out_strategy_2 = (layout2("None", "dp_mp"),)

    in_strategy_3 = (layout("mp", "dp"),)
    out_strategy_3 = (layout("dp", "mp"),)

    strategy_list = ({ "forward": { "input": in_strategy_1, "output": out_strategy_1}},
                     { "forward": { "input": in_strategy_2, "output": out_strategy_2}},
                     { "forward": { "input": in_strategy_3, "output": out_strategy_3}})
    output = run_scenario(
        "Model Parallel (MP)",
        x_layout,
        x_shape=(16, 256),
        strategy_list=strategy_list
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (1, 0)


def test_func_shard_3():
    '''
    Feature: Model parallel in python shard, tp extend ep.
    Description: Test model parallel in python shard with changeable device matrix .
    Expectation: Run success.
    '''
    layout3 = Layout(base_device_matrix3, base_alias_name3, base_rank_list3)
    x_layout = layout3("cp", "ep", "tp")

    in_strategy_1 = (layout3("cp", ("ep", "tp"), "None"),)
    out_strategy_1 = (layout3("cp", ("ep", "tp"), "None"),)

    in_strategy_2 = (layout3("cp", "ep", "tp"),)
    out_strategy_2 = (layout3("cp", "ep", "tp"),)

    in_strategy_3 = (layout3(("cp", "ep"), "None", "tp"),)
    out_strategy_3 = (layout3("cp", "ep", "tp"),)

    strategy_list = ({ "forward": { "input": in_strategy_1, "output": out_strategy_1}},
                     { "forward": { "input": in_strategy_2, "output": out_strategy_2}},
                     { "forward": { "input": in_strategy_3, "output": out_strategy_3}})

    output = run_scenario(
        "Model Parallel (MP)",
        x_layout,
        x_shape=(16, 32, 16),
        strategy_list=strategy_list
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (2, 1, 0)

def test_func_shard_with_bprop():
    '''
    Feature: Model parallel in python shard.
    Description: Test model parallel in python shard with changeable device matrix and bprop .
    Expectation: Run success.
    '''
    layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    layout2 = Layout(base_device_matrix2, base_alias_name2, base_rank_list2)
    x_layout = layout("dp", "None")
    w_layout = layout("None", "None")
    target_layout = layout("dp", "None")

    in_strategy_1 = (layout("None", "None"),)
    out_strategy_1 = None

    in_strategy_2 = (layout2("dp_mp", "None"),)
    out_strategy_2 = (layout2("None", "dp_mp"),)

    in_strategy_3 = (layout("mp", "dp"),)
    out_strategy_3 = (layout("dp", "None"),)

    strategy_list = ({ "forward": { "input": in_strategy_1, "output": out_strategy_1}},
                     { "forward": { "input": in_strategy_2, "output": out_strategy_2}},
                     { "forward": { "input": in_strategy_3, "output": out_strategy_3}})
    run_scenario_with_bprop(
        x_layout,
        w_layout,
        target_layout,
        strategy_list=strategy_list
    )


def test_linear_model_parallel():
    '''
    Feature: Linear model parallel in python shard.
    Description: Test linear model parallel in python shard.
    Expectation: Run success.
    '''
    class Net(nn.Cell):
        """Net composed of several ReLUs"""
        def __init__(self):
            super().__init__()
            self.matmul_net = MatMulNet()
            self.relu_net = ReLUNet()

        def construct(self, x, w):
            out = self.matmul_net(x, w)
            out = self.relu_net(out)
            return out

    D.init()
    np.random.seed(1)
    m, k, n = 256, 128, 64
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(k, n).astype(np.float32))

    # Standalone
    standalone_net = Net()
    standalone_output = standalone_net(x, w)

    # Parallel
    layout = Layout(base_device_matrix, base_alias_name)
    parallel_net = Net()
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    x_local = global_to_local(x, x_layout)
    w_local = global_to_local(w, w_layout)


    strategy_list = { "forward": { "relu_net.input": (layout("dp", "None"),) } }
    shard(parallel_net, sharding_plan=strategy_list)
    parallel_output = parallel_net(x_local, w_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_linear_sequence_parallel():
    '''
    Feature: Linear sequence parallel in python shard.
    Description: Test linear sequence parallel in python shard.
    Expectation: Run success.
    '''
    class Net(nn.Cell):
        """Net composed of several ReLUs"""
        def __init__(self):
            super().__init__()
            self.matmul_net = MatMulNet()
            self.relu_net = ReLUNet()

        def construct(self, x, w):
            out = self.matmul_net(x, w)
            out = self.relu_net(out)
            return out

    D.init()
    np.random.seed(1)
    m, k, n = 256, 128, 64
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(k, n).astype(np.float32))

    # Standalone
    standalone_net = Net()
    standalone_output = standalone_net(x, w)

    # Parallel
    layout = Layout(base_device_matrix, base_alias_name)
    parallel_net = Net()
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    x_local = global_to_local(x, x_layout)
    w_local = global_to_local(w, w_layout)
    strategy_list = { "forward": { "relu_net.input": (layout("dp", "mp"),) } }
    shard(parallel_net, sharding_plan=strategy_list)
    parallel_output = parallel_net(x_local, w_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


@pytest.mark.parametrize('lazy_init', [True, False])
def test_func_shard_with_parameter_plan(lazy_init):
    '''
    Feature: Cell shard with parameter_plan.
    Description: Test cell shard with parameter_plan in python shard.
    Expectation: Run success.
    '''
    class Net(nn.Cell):
        """Net composed of several ReLUs"""
        def __init__(self):
            super().__init__()
            self.matmul_net = MatMulNet()
            self.relu_net = ReLUNet()
            self.weight = Parameter(initializer('ones', shape=(k, n), dtype=mstype.float32), name="weight")

        def construct(self, x):
            out = self.matmul_net(x, self.weight)
            out = self.relu_net(out)
            return out

    D.init()
    np.random.seed(1)
    m, k, n = 256, 128, 64
    x = Tensor(np.random.randn(m, k).astype(np.float32))

    # Standalone
    standalone_net = Net()
    standalone_output = standalone_net(x)

    # Parallel
    layout = Layout(base_device_matrix, base_alias_name)
    if lazy_init:
        with nn.utils.no_init_parameters():
            parallel_net = Net()
    else:
        parallel_net = Net()
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    x_local = global_to_local(x, x_layout)
    strategy_list1 = { "forward": { "input": (layout("dp", "mp"),) }, "parameter": { "weight": w_layout} }
    shard(parallel_net, sharding_plan=strategy_list1)
    strategy_list2 = { "forward": { "relu_net.input": (layout("dp", "mp"),) }}
    shard(parallel_net, sharding_plan=strategy_list2)
    init_parameters(parallel_net)
    parallel_output = parallel_net(x_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
