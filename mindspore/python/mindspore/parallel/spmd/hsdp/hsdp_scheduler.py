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
"""HSDP scheduler"""
from enum import auto, Enum
from mindspore import Parameter, Tensor
from mindspore.parallel.spmd.hsdp.hsdp_state import HSDPState
from mindspore.parallel.spmd.hsdp.hsdp_comm import HSDPComm

class OptimizerLevel(Enum):
    """
        Optimizer level:
                - SHARD_OPT:
                  Splitting is performed on optimizer state.
                - SHARD_OPT_GRAD:
                  Splitting is performed on optimizer state, and gradients.
                - SHARD_OPT_GRAD_PARAM:
                  Splitting is performed on optimizer state, gradients and weights.
    """
    SHARD_OPT = auto()
    SHARD_OPT_GRAD = auto()
    SHARD_OPT_GRAD_PARAM = auto()

class HSDPScheduler:
    """HSDPShceduler is used to imply optimizer level."""

    def __init__(self, cell, shard_size, threshold, shard_level, accumulate_grad_step):
        self.cell = cell
        self.shard_size = shard_size
        self.shard_param_threshold = threshold
        self.shard_level = shard_level
        self.is_shard_level1 = (self.shard_level == OptimizerLevel.SHARD_OPT)
        self.no_param_sharded = False
        if accumulate_grad_step > 1:
            self.requires_acc_grad = True
            self.acc_grad_factor = 1.0 / accumulate_grad_step
        else:
            self.requires_acc_grad = False
            self.acc_grad_factor = 1.0
        self.requires_grad_sync = Parameter(Tensor(False), name="hsdp_requires_grad_sync", requires_grad=False)
        self.comm = HSDPComm()
        self.hsdp_state = HSDPState(cell, self.comm, shard_size, threshold,
                                    self.requires_acc_grad, self.is_shard_level1)
        self._register_hsdp_hooks()

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.requires_grad_sync.set_data(Tensor(requires_grad_sync))

    def zero_grads(self):
        """set requires grad sync flag to control gradient sync."""
        if self.requires_acc_grad:
            for hsdp_param in self.hsdp_state.hsdp_params:
                hsdp_param.zero_acc_grad()

    def _register_hsdp_hooks(self):
        """register process hooks."""
        for hsdp_param in self.hsdp_state.hsdp_params:
            hsdp_param.param.register_hook(self._get_hsdp_param_grad_hook(hsdp_param))
        if self.no_param_sharded:
            return

        self.cell.register_forward_pre_hook(self._hsdp_forward_pre_hook)
        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self.cell.register_forward_hook(self._hsdp_forward_hook)
            self.cell.register_backward_pre_hook(self._hsdp_backward_pre_hook)
            self.cell.register_backward_hook(self._hsdp_backward_hook)
        elif self.requires_acc_grad:
            self.cell.register_backward_hook(self._hsdp_acc_backward_hook)
        else:
            self.cell.register_backward_hook(self._hsdp_backward_hook)

    def _hsdp_forward_pre_hook(self, cell, inputs):
        """forward pre hook to unshard parameter for forward process."""
        self.hsdp_state.unshard()

    def _hsdp_forward_hook(self, cell, inputs, outputs):
        """forward hook to shard parameter for saving memory."""
        self.hsdp_state.shard()

    def _hsdp_backward_pre_hook(self, cell, grad_outputs):
        """backward pre hook to unshard parameter for backward process."""
        self.hsdp_state.unshard()

    def _hsdp_backward_hook(self, cell, grad_inputs, grad_outputs):
        """backward hook to shard parameter for optimizer process or saving memory."""
        self.hsdp_state.shard()

    def _hsdp_acc_backward_hook(self, cell, grad_inputs, grad_outputs):
        """backward hook to shard parameter for accumunication grad only when requires_grad_sync is True."""
        if self.requires_grad_sync:
            self.hsdp_state.shard()

    def _get_hsdp_param_unsharded_hook(self, param):
        """get hook for unsharded param."""
        def grad_all_reduce_hook(grad):
            return self.comm.all_reduce(param.unsharded_group_name, grad) * param.dp_mean_factor

        def grad_acc_all_reduce_hook(grad):
            grad = grad * self.acc_grad_factor
            param.acc_grad.set_data(param.acc_grad + grad)
            if self.requires_grad_sync:
                return self.comm.all_reduce(param.unsharded_group_name, param.acc_grad) * param.dp_mean_factor
            return param.acc_grad

        if not self.requires_acc_grad:
            return grad_all_reduce_hook
        return grad_acc_all_reduce_hook

    def _get_hsdp_param_fully_sharded_hook(self, param):
        """get hook for fully sharded param."""
        def grad_reduce_scatter_hook(grad):
            return self.comm.reduce_scatter(param.sharded_group_name, grad) * param.op_mean_factor

        def grad_acc_reduce_scatter_hook(grad):
            grad = grad * self.acc_grad_factor
            param.acc_grad.set_data(param.acc_grad + grad)
            if self.requires_grad_sync:
                return self.comm.reduce_scatter(param.sharded_group_name, param.acc_grad) * param.op_mean_factor
            return param.acc_grad

        def grad_reduce_scatter_acc_hook(grad):
            grad = grad * self.acc_grad_factor
            sliced_grad = self.comm.reduce_scatter(param.sharded_group_name, grad) * param.op_mean_factor
            param.acc_grad.set_data(param.acc_grad + sliced_grad)
            return param.acc_grad

        if not self.requires_acc_grad:
            return grad_reduce_scatter_hook
        if self.is_shard_level1:
            return grad_acc_reduce_scatter_hook
        return grad_reduce_scatter_acc_hook

    def _get_hsdp_param_partial_sharded_hook(self, param):
        """get hook for partial sharded param."""
        def grad_reduce_scatter_hook(grad):
            sliced_grad = self.comm.reduce_scatter(param.sharded_group_name, grad) * param.op_mean_factor
            return self.comm.all_reduce(param.unsharded_group_name, sliced_grad) * param.dp_mean_factor

        def grad_acc_reduce_scatter_hook(grad):
            grad = grad * self.acc_grad_factor
            param.acc_grad.set_data(param.acc_grad + grad)
            if self.requires_grad_sync:
                sliced_grad = self.comm.reduce_scatter(param.sharded_group_name, param.acc_grad) * param.op_mean_factor
                return self.comm.all_reduce(param.unsharded_group_name, sliced_grad) * param.dp_mean_factor
            return param.acc_grad

        def grad_reduce_scatter_acc_hook(grad):
            grad = grad * self.acc_grad_factor
            sliced_grad = self.comm.reduce_scatter(param.sharded_group_name, grad) * param.op_mean_factor
            param.acc_grad.set_data(param.acc_grad + sliced_grad)
            if self.requires_grad_sync:
                return self.comm.all_reduce(param.unsharded_group_name, param.acc_grad) * param.dp_mean_factor
            return param.acc_grad

        if not self.requires_acc_grad:
            return grad_reduce_scatter_hook
        if self.is_shard_level1:
            return grad_acc_reduce_scatter_hook

        return grad_reduce_scatter_acc_hook

    def _get_hsdp_param_grad_hook(self, param):
        """get hook for param gradient process."""
        if not param.sharded:
            return self._get_hsdp_param_unsharded_hook(param)

        if param.fully_sharded:
            return self._get_hsdp_param_fully_sharded_hook(param)

        return self._get_hsdp_param_partial_sharded_hook(param)
