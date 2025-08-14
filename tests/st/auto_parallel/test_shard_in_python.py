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

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore.parallel import Layout


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")

class SimpleNet(nn.Cell):
    """Net with MatMul and ReLU"""
    def construct(self, x, w):
        x = ms.ops.MatMul()(x, w)
        x = x * 2
        x = ms.mint.nn.ReLU()(x)
        x = x / 10
        x = x + 1
        return x

def create_dtensor(data, layout):
    """create_dtensor"""
    tensor = Tensor(data, dtype=ms.float32)
    return tensor.local_to_global(layout)

def print_layout_info(tensor, name):
    """print_layout_info"""
    if hasattr(tensor, 'layout') and tensor.layout is not None:
        layout_dict = tensor.layout.to_dict()
        print(f"{name} Layout:")
        print(f"  device_matrix: {layout_dict['device_matrix']}")
        print(f"  tensor_map: {layout_dict['tensor_map']}")
        print(f"  alias_name: {layout_dict['alias_name']}")
        print(f"  partial: {tensor.layout.partial}")
        print(f"  rank_list: {layout_dict['rank_list'][:8]}...")  # 只显示前8个rank
    else:
        print(f"{name} has no layout information")

def run_scenario(scenario_name, x_layout, w_layout, x_shape, w_shape):
    """run_scenario"""
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print('='*80)

    # Create Data
    np_x = np.random.randn(*x_shape).astype(np.float32)
    np_w = np.random.randn(*w_shape).astype(np.float32)

    # Create Dtensor
    x = create_dtensor(np_x, x_layout)
    w = create_dtensor(np_w, w_layout)
    print_layout_info(x, "Input X")
    print_layout_info(w, "Input W")

    # Create Net
    net = SimpleNet()
    output = net(x, w)
    print_layout_info(output, "Output")

    return output

base_device_matrix = (2, 4)  # dp=2, mp=4
base_alias_name = ("dp", "mp")
base_rank_list = list(range(8))
D.init()

def test_data_parallel():
    '''
    Feature: Data parallel in python shard.
    Description: Test data parallel in python shard.
    Expectation: Run success.
    '''
    x_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout1 = x_layout1("dp", "None")

    w_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout1 = w_layout1("None", "None")

    output = run_scenario(
        "1. Data Parallel (DP)",
        x_layout1,
        w_layout1,
        x_shape=(16, 256),
        w_shape=(256, 512)
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (1, -1)


def test_model_parallel():
    '''
    Feature: Model parallel in python shard.
    Description: Test model parallel in python shard.
    Expectation: Run success.
    '''
    x_layout2 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout2 = x_layout2("None", "None")

    w_layout2 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout2 = w_layout2("None", "mp")

    output = run_scenario(
        "2. Model Parallel (MP)",
        x_layout2,
        w_layout2,
        x_shape=(16, 256),
        w_shape=(256, 512)
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (-1, 0)


def test_hybrid_parallel():
    '''
    Feature: Hybrid parallel in python shard.
    Description: Test hybrid parallel in python shard.
    Expectation: Run success.
    '''
    x_layout3 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout3 = x_layout3("dp", "None")

    w_layout3 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout3 = w_layout3("None", "mp")

    output = run_scenario(
        "3. Hybrid Parallel (DP + MP)",
        x_layout3,
        w_layout3,
        x_shape=(16, 256),  #
        w_shape=(256, 512)
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (1, 0)


def test_tensor_parallel():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    x_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout4 = x_layout4("None", "mp")

    w_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout4 = w_layout4("mp", "None")

    output = run_scenario(
        "4. Tensor Model Parallel (TMP)",
        x_layout4,
        w_layout4,
        x_shape=(16, 256),
        w_shape=(256, 512)
    )
    output = output.reduce_partial()
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (-1, -1)
    assert output.shape == (16, 512)


def test_hybrid_tensor_parallel():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    x_layout5 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout5 = x_layout5("dp", "mp")

    w_layout5 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout5 = w_layout5("mp", "None")

    output = run_scenario(
        "5. Hybrid Tensor Model Parallel (TMP)",
        x_layout5,
        w_layout5,
        x_shape=(16, 256),
        w_shape=(256, 512)
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (1, -1)


def test_multi_shard_tensor_parallel():
    '''
    Feature: Multi shard tensor parallel in python shard.
    Description: Test multi shard tensor parallel in python shard.
    Expectation: Run success.
    '''
    device_matrix = (2, 2, 2)  # dp=2, mp=4
    alias_name = ("dp", "tp", "mp")
    rank_list = list(range(8))
    x_layout6 = Layout(device_matrix, alias_name, rank_list)
    x_layout6 = x_layout6("dp", "tp")

    w_layout6 = Layout(device_matrix, alias_name, rank_list)
    w_layout6 = w_layout6("tp", "mp")

    output = run_scenario(
        "6. Multi shard Tensor Model Parallel (TMP)",
        x_layout6,
        w_layout6,
        x_shape=(16, 256),
        w_shape=(256, 512)
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (2, 0)

def test_multi_shard_one_dim_tensor_parallel():
    '''
    Feature: Multi shard one dim tensor parallel in python shard.
    Description: Test Multi shard one dim tensor parallel in python shard.
    Expectation: Run success.
    '''
    device_matrix = (2, 2, 2)  # dp=2, mp=4
    alias_name = ("dp", "tp", "mp")
    rank_list = list(range(8))
    x_layout7 = Layout(device_matrix, alias_name, rank_list)
    x_layout7 = x_layout7(("dp", "tp"), "None")

    w_layout7 = Layout(device_matrix, alias_name, rank_list)
    w_layout7 = w_layout7("None", "mp")

    output = run_scenario(
        "7. Multi shard in one dim Tensor Model Parallel (TMP)",
        x_layout7,
        w_layout7,
        x_shape=(16, 256),
        w_shape=(256, 512)
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == ((2, 1), 0)
