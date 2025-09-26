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
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.parallel.spmd.hsdp.hsdp_utils import OptimizerLevel, HSDPConfig
from mindspore.parallel.spmd.hsdp.hsdp_state import HSDPState
import mindspore.parallel.spmd.hsdp.hsdp_comm as comm


class HSDPScheduler:
    """HSDPScheduler is used to imply optimizer level."""

    def __init__(self, cell, shard_size, threshold, shard_level, enable_grad_accumulation, grad_scale,
                 reduce_dtype):
        """init hsdp scheduler."""
        self.cell = cell
        self.shard_level = shard_level
        self.no_param_sharded = (shard_size == 1)
        self.use_cell_hook = (ms.get_context("mode") != ms.GRAPH_MODE)
        self.requires_acc_grad = enable_grad_accumulation
        self.grad_scale = grad_scale
        self.reduce_dtype = reduce_dtype
        self.config = HSDPConfig(
            shard_size,
            threshold,
            self.requires_acc_grad,
            shard_level,
            self.use_cell_hook,
            self.reduce_dtype
        )

        self.requires_grad_sync = Parameter(Tensor(False), name="hsdp_requires_grad_sync", requires_grad=False)
        self.hsdp_state = HSDPState(cell, self.config)
        self.forward_prefetch_cells = []
        self.backward_prefetch_cells = []

        if self.use_cell_hook:
            self._register_cell_hooks()
        else:
            self._register_param_hook()

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        if self.use_cell_hook:
            self.requires_grad_sync = requires_grad_sync
        else:
            ops.assign(self.requires_grad_sync, Tensor(requires_grad_sync))

    def zero_grads(self):
        """set requires grad sync flag to control gradient sync."""
        if self.requires_acc_grad:
            for hsdp_param in self.hsdp_state.hsdp_params:
                if not hsdp_param.param.requires_grad:
                    continue
                hsdp_param.zero_acc_grad()

    def _get_param_forward_hook(self, hsdp_param):
        """get param forward hook."""

        def stateless_param_forward_hook(origin_param):
            output, _ = comm.all_gather_into_tensor(origin_param, group=hsdp_param.sharded_group_name)
            return output

        def stateful_param_forward_hook(origin_param):
            if hsdp_param.unsharded_param_available:
                return hsdp_param.unsharded_param

            unshared_data, _ = comm.all_gather_into_tensor(origin_param, group=hsdp_param.sharded_group_name)
            ops.assign(hsdp_param.unsharded_param, unshared_data)
            ops.assign(hsdp_param.unsharded_param_available, Tensor(True))
            return hsdp_param.unsharded_param

        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            return stateless_param_forward_hook
        return stateful_param_forward_hook

    def _register_param_hook(self):
        """register param forward and grad hook."""
        for hsdp_param in self.hsdp_state.hsdp_params:
            if not hsdp_param.sharded:
                hsdp_param.param.register_hook(self._get_param_grad_hook(hsdp_param))
            else:
                hsdp_param.param.register_hsdp_hook(self._get_param_forward_hook(hsdp_param),
                                                    self._get_param_backward_hook(hsdp_param))

    def _register_cell_hooks(self):
        """register cell process hooks."""
        for hsdp_param in self.hsdp_state.hsdp_params:
            if not hsdp_param.param.requires_grad:
                continue
            hsdp_param.param.register_hook(self._get_param_grad_hook(hsdp_param))
        if self.no_param_sharded:
            return

        self.cell.register_forward_pre_hook(self._hsdp_forward_pre_hook)
        self.cell.register_backward_pre_hook(self._hsdp_backward_pre_hook)
        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self.cell.register_forward_hook(self._hsdp_forward_hook)
            self.cell.register_backward_hook(self._hsdp_backward_hook)
        elif self.requires_acc_grad:
            self.cell.register_backward_hook(self._hsdp_acc_backward_hook)
        else:
            self.cell.register_backward_hook(self._hsdp_backward_hook)

    def _hsdp_forward_pre_hook(self, cell, inputs):
        """forward pre hook to unsharded parameter for forward process."""
        self.hsdp_state.unshard()
        for prefetch_cell in self.forward_prefetch_cells:
            prefetch_cell.hsdp_scheduler.hsdp_state.unshard()

    def _hsdp_forward_hook(self, cell, inputs, outputs):
        """forward hook to shard parameter for saving memory."""
        self.hsdp_state.shard()

    def _hsdp_backward_pre_hook(self, cell, grad_outputs):
        """backward pre hook to unsharded parameter for backward process."""
        self.hsdp_state.unshard()
        for prefetch_cell in self.backward_prefetch_cells:
            prefetch_cell.hsdp_scheduler.hsdp_state.unshard()

    def _hsdp_backward_hook(self, cell, grad_inputs, grad_outputs):
        """backward hook to shard parameter for optimizer process or saving memory."""
        self.hsdp_state.shard()

    def _hsdp_acc_backward_hook(self, cell, grad_inputs, grad_outputs):
        """backward hook to shard parameter for grad accumulation when requires_grad_sync is True."""
        if self.requires_grad_sync:
            self.hsdp_state.shard()

    def _cast_hook(self, hook, grad):
        if self.reduce_dtype is None:
            return hook(grad)
        origin_dtype = ops.dtype(grad)
        grad_cast = ops.cast(grad, self.reduce_dtype)
        output = hook(grad_cast)
        output = ops.cast(output, origin_dtype)
        return output

    def _get_grad_scale_hook(self, param, grad_hook):
        def scale_hook(grad):
            output = self._cast_hook(grad_hook, grad)
            scale_output = output * self.grad_scale
            param.grad = scale_output
            return scale_output
        return scale_hook

    def _get_hsdp_param_single_node_hook(self, hsdp_param):
        """get hook for unsharded param with single node."""
        def grad_dummy_hook(grad):
            output = grad * self.grad_scale
            hsdp_param.param.grad = output
            return output

        def grad_hook(grad):
            ops.assign_add(hsdp_param.acc_grad, grad)
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            return grad_dummy_hook
        return self._get_grad_scale_hook(hsdp_param.param, grad_hook)

    def _get_hsdp_param_unsharded_hook(self, hsdp_param):
        """get hook for unsharded param."""
        def grad_all_reduce_hook(grad):
            output, _ = comm.all_reduce(grad, group=hsdp_param.unsharded_group_name)
            return output

        def grad_acc_all_reduce_hook(grad):
            ops.assign_add(hsdp_param.acc_grad, grad)
            if self.requires_grad_sync:
                output, _ = comm.all_reduce(hsdp_param.acc_grad, group=hsdp_param.unsharded_group_name)
                return output
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            grad_hook = grad_all_reduce_hook
        else:
            grad_hook = grad_acc_all_reduce_hook
        return self._get_grad_scale_hook(hsdp_param.param, grad_hook)

    def _get_hsdp_param_fully_sharded_hook(self, hsdp_param):
        """get hook for fully sharded param."""
        def grad_reduce_scatter_hook(grad):
            output, _ = comm.reduce_scatter_tensor(grad, group=hsdp_param.sharded_group_name)
            return output

        def grad_acc_reduce_scatter_hook(grad):
            ops.assign_add(hsdp_param.acc_grad, grad)
            if self.requires_grad_sync:
                output, _ = comm.reduce_scatter_tensor(hsdp_param.acc_grad, group=hsdp_param.sharded_group_name)
                return output
            return hsdp_param.acc_grad

        def grad_reduce_scatter_acc_hook(grad):
            output, _ = comm.reduce_scatter_tensor(grad, group=hsdp_param.sharded_group_name)
            ops.assign_add(hsdp_param.acc_grad, output)
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            grad_hook = grad_reduce_scatter_hook
        elif self.shard_level == OptimizerLevel.SHARD_OPT:
            grad_hook = grad_acc_reduce_scatter_hook
        else:
            grad_hook = grad_reduce_scatter_acc_hook
        return self._get_grad_scale_hook(hsdp_param.param, grad_hook)

    def _get_hsdp_param_partial_sharded_hook(self, hsdp_param):
        """get hook for partial sharded param."""
        def grad_reduce_scatter_hook(grad):
            output, _ = comm.reduce_scatter_tensor(grad, group=hsdp_param.sharded_group_name)
            sliced_grad, _ = comm.all_reduce(output, group=hsdp_param.unsharded_group_name)
            return sliced_grad

        def grad_acc_reduce_scatter_hook(grad):
            ops.assign_add(hsdp_param.acc_grad, grad)
            if self.requires_grad_sync:
                output, _ = comm.reduce_scatter_tensor(hsdp_param.acc_grad, group=hsdp_param.sharded_group_name)
                sliced_grad, _ = comm.all_reduce(output, group=hsdp_param.unsharded_group_name)
                return sliced_grad
            return hsdp_param.acc_grad

        def grad_reduce_scatter_acc_hook(grad):
            output, _ = comm.reduce_scatter_tensor(grad, group=hsdp_param.sharded_group_name)
            ops.assign_add(hsdp_param.acc_grad, output)
            if self.requires_grad_sync:
                output, _ = comm.all_reduce(hsdp_param.acc_grad, group=hsdp_param.unsharded_group_name)
                return output
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            grad_hook = grad_reduce_scatter_hook
        elif self.shard_level == OptimizerLevel.SHARD_OPT:
            grad_hook = grad_acc_reduce_scatter_hook
        else:
            grad_hook = grad_reduce_scatter_acc_hook
        return self._get_grad_scale_hook(hsdp_param.param, grad_hook)

    def _get_param_grad_hook(self, hsdp_param):
        """get hook for param gradient process."""
        if not hsdp_param.sharded:
            if hsdp_param.dp_size == 1:
                return self._get_hsdp_param_single_node_hook(hsdp_param)
            return self._get_hsdp_param_unsharded_hook(hsdp_param)

        if hsdp_param.fully_sharded:
            return self._get_hsdp_param_fully_sharded_hook(hsdp_param)

        return self._get_hsdp_param_partial_sharded_hook(hsdp_param)

    def _get_param_backward_hook(self, hsdp_param):
        """get hook for param backward process."""
        grad_hook = self._get_param_grad_hook(hsdp_param)
        def backward_hook(grad):
            ops.assign(hsdp_param.unsharded_param_available, Tensor(False))
            return grad_hook(grad)

        def backward_acc_grad_hook(grad):
            if self.requires_grad_sync:
                ops.assign(hsdp_param.unsharded_param_available, Tensor(False))
            return grad_hook(grad)

        if self.requires_acc_grad:
            return backward_acc_grad_hook
        return backward_hook

    def set_forward_prefetch_cells(self, hsdp_cell_list):
        """set forward prefetch cells."""
        self.forward_prefetch_cells = hsdp_cell_list

    def set_backward_prefetch_cells(self, hsdp_cell_list):
        """set backward prefetch cells."""
        self.backward_prefetch_cells = hsdp_cell_list
