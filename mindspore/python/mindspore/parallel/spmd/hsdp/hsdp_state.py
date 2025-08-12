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
"""HSDP cell state"""
from mindspore.parallel.spmd.hsdp.hsdp_param import HSDPParam
from mindspore import Parameter, Tensor

class HSDPState:
    """HSDP state for cell"""
    def __init__(self, cell, comm, shard_size, threshold, requires_acc_grad, is_shard_level1):
        self.cell = cell
        self.comm = comm
        self.shard_size = shard_size
        self.threshold = threshold
        self.requires_acc_grad = requires_acc_grad
        self.is_shard_level1 = is_shard_level1
        self.hsdp_params = []
        self.sharded_hsdp_params = []
        self._init_hsdp_params()
        self.is_shard = Parameter(Tensor(True), name="hsdp_requires_grad_sync", requires_grad=False)

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        cells = self.cell.cells_and_names()
        for _, sub_cell in cells:
            params = sub_cell._params.items() #pylint: disable=W0212
            for param_name, param in params:
                if not param.requires_grad:
                    continue
                if hasattr(param, "has_hsdp_param"):
                    continue
                hsdp_param = HSDPParam(sub_cell, param_name, param, self)
                param.has_hsdp_param = True
                self.hsdp_params.append(hsdp_param)
                if hsdp_param.sharded:
                    self.sharded_hsdp_params.append(hsdp_param)

    def shard(self):
        """change parameters to sharded state"""
        if self.is_shard:
            return
        for param in self.sharded_hsdp_params:
            param.to_sharded()
        self.is_shard.set_data(Tensor(True))

    def unshard(self):
        """change parameters to unsharded state"""
        if not self.is_shard:
            return
        for param in self.sharded_hsdp_params:
            param.to_unsharded()
        self.is_shard.set_data(Tensor(False))
