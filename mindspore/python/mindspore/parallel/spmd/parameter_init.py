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
"""Parameter init"""

def init_parameters(cell):
    r"""
        init parameters.

        Args:
            cell(Cell): The cell to init parameters.
        Raises:
            ValueError: If the `cell` is not a cell.
    """
    from mindspore.nn.cell import Cell
    if not isinstance(cell, Cell):
        raise ValueError("cell's type must be Cell but got {}.".format(type(cell)))
    for param in cell.get_parameters(expand=True):
        if not param.has_init:
            continue
        init_data = param.init_mode.init_data()
        param.init_mode = None
        param.init = None
        param.set_data(init_data)
    return cell
