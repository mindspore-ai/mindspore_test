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
from mindspore import nn, Tensor, mint
from mindspore.parallel import Layout

D.init()
ms.set_context(pynative_synchronize=True)


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


class SimpleIndexSelectNet(nn.Cell):
    """Net with Index Select"""

    def construct(self, params, axis, indices):
        out = mint.index_select(params, axis, indices)
        return out


class SimpleGatherNet(nn.Cell):
    """Net with Gather"""

    def construct(self, params, axis, indices):
        out = mint.gather(params, axis, indices)
        return out


def create_dtensor(data, layout):
    """create_dtensor"""
    tensor = Tensor(data, dtype=ms.float32)
    return tensor.local_to_global(layout)


def print_layout_info(tensor, name):
    """print_layout_info"""
    if hasattr(tensor, "layout") and tensor.layout is not None:
        layout_dict = tensor.layout.to_dict()
        print(f"{name} Layout:")
        print(f"  device_matrix: {layout_dict['device_matrix']}")
        print(f"  tensor_map: {layout_dict['tensor_map']}")
        print(f"  alias_name: {layout_dict['alias_name']}")
        print(f"  partial: {tensor.layout.partial}")
        print(f"  rank_list: {layout_dict['rank_list'][:8]}...")  # 只显示前8个rank
    else:
        print(f"{name} has no layout information")


def run_scenario(scenario_name, net, params_layout, indices_layout, params_shape, indices_shape, axis):
    """run_scenario"""
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    # Create Dtensor
    params = Tensor(
        np.random.randn(*params_shape).astype(np.float32), dtype=ms.float32
    ).local_to_global(params_layout)
    indices = Tensor(np.ones(indices_shape), ms.int32).local_to_global(indices_layout)

    print_layout_info(params, "Input Params")
    print_layout_info(indices, "Input Indices")

    # Create Net
    net = net()
    output = net(params, axis, indices)
    print_layout_info(output, "Output")

    return output


def test_index_select_parallel():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    p_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    p_layout = p_layout("None", "a")

    i_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    i_layout = i_layout("b")

    output = run_scenario(
        "1. [index_select] Params shard: (1, 2), Indices shard: (4)",
        SimpleIndexSelectNet,
        p_layout,
        i_layout,
        params_shape=(16, 256),
        indices_shape=(512),
        axis=0,
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (0, 1)


def test_gather_parallel():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    p_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    p_layout = p_layout("None", "a")

    i_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    i_layout = i_layout("b", "a")

    output = run_scenario(
        "1. [gather] Params shard: (1, 2), Indices shard: (4, 2)",
        SimpleGatherNet,
        p_layout,
        i_layout,
        params_shape=(16, 512),
        indices_shape=(32, 256),
        axis=0,
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (0, 1)
