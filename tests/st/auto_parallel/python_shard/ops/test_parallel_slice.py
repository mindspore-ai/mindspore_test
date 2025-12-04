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
"""test parallel slice op"""
# pylint: disable=W0622
import pytest
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore.parallel import Layout
from mindspore.ops.function import slice
from mindspore.parallel._tensor import _load_tensor_by_layout

D.init()

def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


class SimpleNet(nn.Cell):
    """Net with MatMul and ReLU"""

    def construct(self, x, w, begin, end):
        x = slice(x, begin, end)
        x = x + w
        return x


class AllGatherNet(nn.Cell):
    """Net with MatMul and ReLU"""

    def construct(self, x):
        return x


def create_dtensor(data, layout):
    """create_dtensor"""
    return data.local_to_global(layout)


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


def run_scenario_parallel(scenario_name, x_layout, w_layout, begin, end, x, w):
    """run_scenario"""
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print('='*80)

    # Create Dtensor
    x = create_dtensor(x, x_layout)
    w = create_dtensor(w, w_layout)
    print_layout_info(x, "Input X")
    print_layout_info(w, "Input W")

    dev_mat = Layout((8,), ("all_dev",))
    tensor_map = ["None"] * len(x.shape)
    # Create Net
    net = SimpleNet()
    allgather_net = AllGatherNet()
    allgather_net.shard(in_strategy=(dev_mat(*tensor_map),))
    output = net(x, w, begin, end)
    output = allgather_net(output)
    print_layout_info(output, "Output")

    return output


def run_scenario_standalone(scenario_name, begin, end, x, w):
    """run_scenario"""
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print('='*80)

    # Create Dtensor
    print_layout_info(x, "Input X")
    print_layout_info(w, "Input W")

    # Create Net
    net = SimpleNet()
    output = net(x, w, begin, end)
    print_layout_info(output, "Output")

    return output


def create_ms_data(layout, local_shape):
    seed = np.random.RandomState(seed=12)
    global_shape = layout.get_global_shape(local_shape)
    global_tensor = Tensor(seed.random(global_shape), ms.float32)
    layout_dict = layout.to_dict()
    layout_tuple = tuple([layout_dict["device_matrix"], layout_dict["tensor_map"],
                          local_shape, False, True, False, global_shape])
    local_tensor = _load_tensor_by_layout(global_tensor, layout_tuple, D.get_rank())
    return global_tensor, local_tensor


def test_slice_1():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    base_device_matrix = (2, 4)  # dp=2, mp=4
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout4 = x_layout4("dp", "None", "mp", "None")

    w_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout4 = w_layout4("dp", "None", "mp", "None")

    local_shape_x = (16, 256, 8, 256)
    local_shape_w = (16, 256 // 2, 8, 256)
    # Create Data
    x_standalone, x_parallel = create_ms_data(x_layout4, local_shape_x)
    w_standalone, w_parallel = create_ms_data(w_layout4, local_shape_w)

    output_parallel = run_scenario_parallel(
        "1. Slice Parallel Op",
        x_layout4,
        w_layout4,
        begin=(0, 0, 0, 0),
        end=(32, 256 // 2, 32, 256),
        x=x_parallel,
        w=w_parallel
    )
    output_standalone = run_scenario_standalone(
        "2. Slice Standalone Op",
        begin=(0, 0, 0, 0),
        end=(32, 256 // 2, 32, 256),
        x=x_standalone,
        w=w_standalone
    )
    output_layout = output_parallel.layout
    assert output_layout is not None
    np.allclose(output_standalone.asnumpy(), output_standalone.asnumpy())


def test_slice_2():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    base_device_matrix = (2, 4)  # dp=2, mp=4
    base_alias_name = ("dp", "mp")
    base_rank_list = list(range(8))

    x_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout4 = x_layout4("dp", "None", "mp", "None")

    w_layout4 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout4 = w_layout4("dp", "None", "mp", "None")

    local_shape_x = (16, 256, 8, 256)
    local_shape_w = (16, 256 // 2, 8, 256)
    # Create Data
    _, x_parallel = create_ms_data(x_layout4, local_shape_x)
    _, w_parallel = create_ms_data(w_layout4, local_shape_w)

    with pytest.raises(ValueError):
        _ = run_scenario_parallel(
            "1. Slice Parallel Op",
            x_layout4,
            w_layout4,
            begin=(0, 0, 0, 0),
            end=(16, 256 // 2, 32, 256),
            x=x_parallel,
            w=w_parallel
        )
