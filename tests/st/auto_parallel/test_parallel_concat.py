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


class SimpleNet(nn.Cell):
    """Net with Concat"""

    def construct(self, tensors, dim):
        out = mint.cat(tensors, dim)
        return out


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


def run_scenario(scenario_name, tensor_1_layout, tensor_2_layout,
                 tensor_1_shape, tensor_2_shape, dim=0):
    """run_scenario"""
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print("=" * 80)

    # Create Dtensor
    tensor_1 = Tensor(
        np.random.randn(*tensor_1_shape).astype(np.float32), dtype=ms.float32
    ).local_to_global(tensor_1_layout)
    tensor_2 = Tensor(
        np.random.randn(*tensor_2_shape).astype(np.float32), dtype=ms.float32
    ).local_to_global(tensor_2_layout)
    print_layout_info(tensor_1, "Input tensor_1")
    print_layout_info(tensor_2, "Input tensor_2")

    # Create Net
    net = SimpleNet()
    output = net((tensor_1, tensor_2), dim)
    print_layout_info(output, "Output")

    return output


def test_concat_parallel():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    base_device_matrix = (2, 4, 1)
    base_alias_name = ("a", "b", "c")
    base_rank_list = list(range(8))

    t1_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)(
        "a", "b", "c"
    )
    t2_layout = Layout(base_device_matrix, base_alias_name, base_rank_list)(
        "None", "b", "c"
    )

    output = run_scenario(
        "Concat Parallel", t1_layout, t2_layout, (16, 256), (32, 256), 0
    )
    output_layout = output.layout
    assert output_layout is not None
    output_layout_dict = output_layout.to_dict()
    assert output_layout_dict["tensor_map"] == (2, 1, 0)
