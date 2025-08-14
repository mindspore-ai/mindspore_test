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
"""
Distributed implementation for MatMul operator.
"""

from mindspore.parallel import Layout
from .parallel_ops import DistributedOp


class MatMulExtDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""
    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for MatMul operator.

        MatMul: output = x @ w

        Rules:
        1. Batch dimensions should have same layout
        2. Contracting dimensions should have same layout
        3. Output dimensions inherit layouts from non-contracting dimensions

        Args:
            x_layout (Layout): Layout of input x
            w_layout (Layout): Layout of input w

        Returns:
            tuple: Layout for output tensor
        """
        if len(layouts) != 2:
            raise ValueError(f"MatMul layout length is not 2, but {len(layouts)}")
        x_layout = layouts[0]
        w_layout = layouts[1]
        if not x_layout or not w_layout:
            raise ValueError(f"x_layout : {x_layout}, w_layout : {w_layout}")
        x_device_matrix = x_layout.device_matrix
        w_device_matrix = w_layout.device_matrix
        if x_device_matrix != w_device_matrix:
            raise ValueError("MatMul inputs must have same device_matrix")

        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map
        contract_dim = len(x_map) - 1
        w_contract_dim = len(w_map) - 2
        if x_map[contract_dim] != w_map[w_contract_dim]:
            raise ValueError(f"Contracting dimensions must have same layout. "
                             f"Got {x_map[contract_dim]} and {w_map[w_contract_dim]}")

        output_dim = len(w_map) - 1
        output_map = x_map[:-1] + (w_map[output_dim],)

        output_layout = Layout(
            device_matrix=x_layout.device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        out_layout = output_layout(*output_map)

        # Set partial status
        if x_map[contract_dim] != "None":
            if isinstance(x_map[contract_dim], tuple):
                for axis in x_map[contract_dim]:
                    out_layout.set_partial_by_dev_axis(axis, 'sum')
            else:
                out_layout.set_partial_by_dev_axis(x_map[contract_dim], 'sum')

        return out_layout


class MatMulDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""
    def infer_layout(self, input_layouts, extra_args):
        """
        Infer output layout for MatMul operator.

        MatMul: output = x @ w, with possible transpose

        Args:
            input_layouts (tuple): Layouts of input tensors (x_layout, w_layout)
            extra_args (tuple): Additional arguments (transpose_a, transpose_b)

        Returns:
            Layout: Layout for output tensor
        """
        if len(input_layouts) < 2:
            raise ValueError("MatMul requires at least two input layouts")

        x_layout, w_layout = input_layouts[:2]

        if len(extra_args) != 2:
            raise ValueError("MatMul requires two transpose input")
        transpose_a, transpose_b = extra_args[0], extra_args[1]

        x_dict = x_layout.to_dict()
        w_dict = w_layout.to_dict()

        if x_dict["device_matrix"] != w_dict["device_matrix"]:
            raise ValueError("MatMul inputs must have same device_matrix")

        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        # Determine contracting dimensions based on transpose flags
        if transpose_a:
            x_input_dim = len(x_map) - 1
            x_contract_dim = len(x_map) - 2  # Second to last dimension
        else:
            x_input_dim = len(x_map) - 2
            x_contract_dim = len(x_map) - 1  # Last dimension

        if transpose_b:
            w_output_dim = len(w_map) - 2
            w_contract_dim = len(w_map) - 1  # Last dimension
        else:
            w_output_dim = len(w_map) - 1
            w_contract_dim = len(w_map) - 2  # Second to last dimension

        # Validate contracting dimensions
        if x_map[x_contract_dim] != w_map[w_contract_dim]:
            raise ValueError(f"Contracting dimensions must have same layout. "
                             f"Got {x_map[x_contract_dim]} and {w_map[w_contract_dim]}")

        # Create output layout
        output_layout = Layout(
            device_matrix=x_layout.device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map = list(x_map[:-2]) + [x_map[x_input_dim]] + [w_map[w_output_dim]]
        output_layout = output_layout(*output_map)

        # Set partial status
        if x_map[x_contract_dim] != "None":
            if isinstance(x_map[x_contract_dim], tuple):
                for axis in x_map[x_contract_dim]:
                    output_layout.set_partial_by_dev_axis(axis, 'sum')
            else:
                output_layout.set_partial_by_dev_axis(x_map[x_contract_dim], 'sum')

        return output_layout
