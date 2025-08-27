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

import pytest
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore.parallel import Layout
import mindspore.ops as ops

D.init()
ms.set_context(pynative_synchronize=True)


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


class SimpleNet(nn.Cell):
    """Net with Embedding"""

    def construct(self, x, w):
        x = ops.embedding(x, w, None, None, 2.0, False)
        x = x * 2
        x = ms.mint.nn.ReLU()(x)
        x = x + 1
        return x


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
        print(f"  rank_list: {layout_dict['rank_list'][:8]}...")
    else:
        print(f"{name} has no layout information")


def run_scenario(scenario_name, x_layout, w_layout, x_shape, w_shape):
    """run_scenario"""
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    # Create Dtensor
    x = Tensor(np.ones(x_shape), ms.int32).local_to_global(x_layout)
    w = Tensor(
        np.random.randn(*w_shape).astype(np.float32), dtype=ms.float32
    ).local_to_global(w_layout)

    print_layout_info(x, "Input X")
    print_layout_info(w, "Input W")

    # Create Net
    net = SimpleNet()
    output = net(x, w)
    print_layout_info(output, "Output")

    return output


def test_parallel_1():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (2, 4)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None")

    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("a", "b")

    with pytest.raises(ValueError):
        output = run_scenario(
            "1. Cut weight (2, 4)",
            x_layout,
            w_layout,
            x_shape=(16, 256),
            w_shape=(32, 8),
        )
        assert output.layout is not None
        output_layout_dict = output.layout.to_dict()
        assert output_layout_dict["device_matrix"] == base_device_matrix
        assert output_layout_dict["tensor_map"] == (1, -1, 0)


def test_parallel_2():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (8, 1)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None")

    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("a", "b")

    with pytest.raises(ValueError):
        output = run_scenario(
            "2. Cut weight (8, 1)",
            x_layout,
            w_layout,
            x_shape=(16, 256),
            w_shape=(32, 8),
        )
        assert output.layout is not None
        output_layout_dict = output.layout.to_dict()
        assert output_layout_dict["device_matrix"] == base_device_matrix
        assert output_layout_dict["tensor_map"] == (1, -1, 0)


def test_parallel_3():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (1, 8)
    base_alias_name = ("a", "b")
    base_rank_list = list(range(8))

    x_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout = x_layout("None", "None")

    w_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout = w_layout("None", "b")

    output = run_scenario(
        "3. Cut weight (1, 8)",
        x_layout,
        w_layout,
        x_shape=(16, 256),
        w_shape=(32, 8),
    )
    assert output.layout is not None
    output_layout_dict = output.layout.to_dict()
    assert output_layout_dict["device_matrix"] == base_device_matrix
    assert output_layout_dict["tensor_map"] == (-1, -1, 0)
