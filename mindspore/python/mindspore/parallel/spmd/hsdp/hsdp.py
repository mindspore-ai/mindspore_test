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
"""hybrid shard data parallel interface"""
from mindspore.parallel.spmd.hsdp.hsdp_scheduler import HSDPScheduler, OptimizerLevel

origin_class_to_extend_class = {}
optimizer_level_map = {
    "level1": OptimizerLevel.SHARD_OPT,
    "level2": OptimizerLevel.SHARD_OPT_GRAD,
    "level3": OptimizerLevel.SHARD_OPT_GRAD_PARAM,
}

class HSDPCell:
    """
    The hsdp block of neural networks with hsdp interface.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def hsdp_init(self, cell, shard_size, threshold, optimizer_level, accumulate_grad_step):
        """init hsdp scheduler."""
        self.hsdp_scheduler = HSDPScheduler(cell, shard_size, threshold, optimizer_level, accumulate_grad_step)

    def set_requires_grad_sync(self, requires_grad_sync):
        r"""
            set requires grad sync flag.
            Args:
                requires_grad_sync(bool): requires_grad_sync is used to control gradient sync process.
        """
        if not hasattr(self, "hsdp_scheduler"):
            return
        self.hsdp_scheduler.set_requires_grad_sync(requires_grad_sync)
        for sub_cell in self.cells():
            if isinstance(sub_cell, HSDPCell):
                sub_cell.set_requires_grad_sync(requires_grad_sync)

    def zero_grads(self):
        """zero accumunication grads"""
        if not hasattr(self, "hsdp_scheduler"):
            return
        self.hsdp_scheduler.zero_grads()
        for sub_cell in self.cells():
            if isinstance(sub_cell, HSDPCell):
                sub_cell.zero_grads()

def _extend_cell_with_hsdp_interface(cell):
    """extend Cell with HSDPCell interface"""
    origin_class = cell.__class__
    extend_class = origin_class_to_extend_class.get(origin_class, None)
    if extend_class is None:
        extend_class = type(f"HSDP{origin_class.__name__}", (HSDPCell, origin_class), {})
        origin_class_to_extend_class[origin_class] = extend_class
    cell.__class__ = extend_class

def hsdp(cell, shard_size=1, threshold=64, optimizer_level="level1", accumulate_grad_step=1):
    r"""
        apply hybrid sharded data parallel.

        Args:
            shard_size (int, optional): Set the optimizer weight shard group size if you want to specific the
                maximum group size across devices when the parallel optimizer is
                enabled. The numerical range can be (0, device_num]. Default value is 1,
                which means the optimizer weight is not sharded.
            threshold (int, optional): Set the threshold of parallel optimizer. When parallel optimizer is
                enabled, parameters with size smaller than this threshold will not be
                sharded across the devices. Parameter size = shape[0] \* ... \*
                shape[n] \* size(dtype). Non-negative. Unit: KB. Default: 64.
            optimizer_level (str, optional): optimizer_level configuration is used to specify
                the splitting level for optimizer sharding. It is important to note that the implementation
                of optimizer sharding in static graph is inconsistent with dynamic graph like megatron,
                but the memory optimization effect is the same.
                It must be one of [ ``level1``, ``level2``, ``level3`` ]. Default: ``level1``.

                - level1:
                  Splitting is performed on weights and optimizer state.
                - level2:
                  Splitting is performed on weights, optimizer state, and gradients.
                - level3:
                  Splitting is performed on weights, optimizer state,
                  gradients, additionally, before the backward pass, the weights are further applied with
                  allgather communication to release the memory used by the forward pass allgather.
            accumulate_grad_step (int, optional): Set the accumulate grad step.

        Raises:
            ValueError: If the `shard_size` is not a positive integer.
            ValueError: If `threshold` is not a positive integer or 0.
            ValueError: If `optimizer_level` is not one of the [ ``level1``, ``level2``, ``level3`` ].
            ValueError: If `accumulate_grad_step` is not a positive integer or 0.
        """
    if not isinstance(shard_size, int) or (shard_size <= 0 and shard_size != -1):
        raise ValueError("shard_size must be a positive integer, but got {}.".format(shard_size))
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError("threshold must be a positive integer or 0, but got {}.".format(threshold))
    if optimizer_level not in ["level1", "level2", "level3"]:
        raise ValueError("Optimizer level should in ['level1', 'level2', 'level3'], but got {}"
                         .format(optimizer_level))
    optimizer_level = optimizer_level_map.get(optimizer_level)
    if not isinstance(accumulate_grad_step, int) or accumulate_grad_step < 0:
        raise ValueError("accumulate_grad_step must be a positive integer or 0, but got {}."
                         .format(accumulate_grad_step))
    _extend_cell_with_hsdp_interface(cell)
    cell.hsdp_init(cell, shard_size, threshold * 1024, optimizer_level, accumulate_grad_step)
    return cell
