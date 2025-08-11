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

class MatMulDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""
    def __init__(self):
        super().__init__("MatMulExt")

    def infer_layout(self, x_layout, w_layout):
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
        x_dict = x_layout.to_dict()
        w_dict = w_layout.to_dict()
        if x_dict["device_matrix"] != w_dict["device_matrix"]:
            raise ValueError("MatMul inputs must have same device_matrix")

        x_map = x_dict["tensor_map"]
        w_map = w_dict["tensor_map"]
        x_aliases = x_dict["alias_name"]
        contract_dim = len(x_map) - 1
        w_contract_dim = len(w_map) - 2
        if x_map[contract_dim] != w_map[w_contract_dim]:
            raise ValueError(f"Contracting dimensions must have same layout. "
                             f"Got {x_map[contract_dim]} and {w_map[w_contract_dim]}")

        output_map = ()
        for map_id in x_map[:-1]:
            if isinstance(map_id, tuple):
                output_map_map = ()
                for map_id_id in map_id:
                    if map_id_id < 0:
                        output_map_map += ("None",)
                    else:
                        output_map_map += (x_aliases[len(x_aliases) - 1 - map_id_id],)
                output_map += (output_map_map,)
                continue
            if map_id < 0:
                output_map += ("None",)
                continue
            dev_dim = len(x_aliases) - 1 - map_id
            output_map += (x_aliases[dev_dim],)

        output_dim = len(w_map) - 1
        if isinstance(w_map[output_dim], tuple):
            output_map_map = ()
            for map_id_id in w_map[output_dim]:
                if map_id_id < 0:
                    output_map_map += ("None",)
                else:
                    output_map_map += (x_aliases[len(x_aliases) - 1 - map_id_id],)
            output_map += (output_map_map,)
        elif w_map[output_dim] < 0:
            output_map += ("None",)
        else:
            dev_dim = len(x_aliases) - 1 - w_map[output_dim]
            output_map += (x_aliases[dev_dim],)

        output_layout = Layout(
            device_matrix=x_dict["device_matrix"],
            alias_name=x_aliases,
            rank_list=x_dict["rank_list"]
        )
        out_layout = output_layout(*output_map)
        return out_layout
