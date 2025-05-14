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
""" Distributed data parallel wrapper. """
from contextlib import contextmanager
from typing import Optional
import mindspore.nn as nn
import mindspore.log as logger
from mindspore.common import dtype as mstype
from mindspore.mint.distributed import get_world_size
from mindspore.communication import GlobalComm
from .flatten_grad_buffer import FlattenGradBuffer

__all__ = ["DistributedDataParallel"]


def get_data_parallel_group():
    return GlobalComm.WORLD_COMM_GROUP


def get_data_parallel_world_size():
    return get_world_size()


class DistributedDataParallel(nn.Cell):
    """
    DistributedDataParallel wrapper. DistributedDataParallel allocates contiguous memory buffer for gradients.
    parameters and gradients will be break up into bucekts which is the unit to conduct all-reduce
    communication among data parallel group to overlap communication latency.

    Args:
        module (nn.Cell): The module to be wrapped with DDP.
        process_group: The comm group of data prallel
        bucket_cap_mb (Optional[int]): Size of bucket,default is 25MiB.
        average_in_collective(bool): True means allreduce sum within DP group firstly then scaling with dp size.
        Otherwise scaling local rank grad first and then allreduce sum.
        static_graph(bool):Net's parameter types are not changed during forward passes. For dynamic flow case,
        parameters are changed in forward and backward pass for iteration steps, in this case, it is not static graph.

    Outputs:
        Model wrapped with DistributedDataParallel.

    Examples:
        .. note:
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
        >>>    net = DistributedDataParallel(module=net,
        ...                                 bucket_cap_mb=None,
        ...                                 average_in_collective=True,
        ...                                 static_graph=True)
        >>>    optimizer = AdamW(net.trainable_params(), 1e-4)
        >>>    loss_fn = nn.CrossEntropyLoss()
        ...
        >>>    def forward_fn(data, target):
        >>>        logits = net(data)
        >>>        loss = loss_fn(logits, target)
        >>>        return loss, logits
        ...
        >>>    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
        ...
        >>>    for epoch in range(1):
        >>>        step = 0
        >>>        for image, label in data_set:
        >>>            (loss_value, _), grads = grad_fn(image, label)
        >>>            optimizer(grads)
        >>>            net.zero_grad()
        >>>            step += 1
        >>>            print("epoch: %s, step: %s, loss is %.15f" % (epoch, i, loss_value))
    """

    def __init__(self, module, process_group=None, bucket_cap_mb: Optional[int] = None, find_unused_parameters=False,
                 average_in_collective: bool = False, static_graph=False):
        super(DistributedDataParallel, self).__init__(auto_prefix=False)
        self.bucket_cap_mb = bucket_cap_mb
        self.average_in_collective = average_in_collective
        self.grad_reduce_in_fp32 = False
        self.process_group = process_group

        self.module = module
        self.param_to_buffer = {}
        self.has_buckets_grad_sync = False

        # default is 25MB for each buck
        if bucket_cap_mb is None:
            bucket_cap_mb = 25
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # grads sync with allreduce comm
        self.sync_enabled = True

        bucketed_params = []
        for _, param in self.module.parameters_and_names():
            if not param.requires_grad:
                continue
            param.grad = None
            param.main_grad = None
            bucketed_params.append(param)
            if self.average_in_collective:
                # allreduce to add grads, then to scale grads with dp size
                self.gradient_scaling_factor = 1.0
            else:
                # scale grads with dp size locally, then allreduce to add grads
                data_parallel_world_size = get_data_parallel_world_size()
                self.gradient_scaling_factor = 1.0 / data_parallel_world_size

        # allocate buffer for trained params
        self.buffers = self.allocate_buffers_for_parameters(
            bucketed_params,
            group=get_data_parallel_group()
            if self.process_group is None
            else self.process_group,
            gradient_scaling_factor=self.gradient_scaling_factor,
        )

        # register hook for bucket grad reduce
        self._register_hook_for_params()

        # bucket rebuilding
        self.rebuilt_params_ = []
        self.buffer_iterations = 0
        self.has_bucket_rebuilt = False
        self.static_graph = static_graph
        self.buffer_issued = 0

    def allocate_buffers_for_parameters(self, input_params, group, gradient_scaling_factor):
        """allocate buffers for parameters in different dtype group."""
        param_and_grad_dtype_to_params = {}
        # group all params by parameter's data type and their gradient's data type.
        for param in input_params:
            param_dtype = param.dtype
            grad_dtype = mstype.float32 if self.grad_reduce_in_fp32 else param.dtype
            if (param_dtype, grad_dtype) not in param_and_grad_dtype_to_params:
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = []
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)].append(param)

        buffers = []
        # allocate buffer for each group separately
        for (param_dtype, grad_dtype,), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                FlattenGradBuffer(
                    average_in_collective=self.average_in_collective,
                    param_dtype=param_dtype,
                    grad_dtype=grad_dtype,
                    params=params,
                    data_parallel_group=group,
                    bucket_size=self.bucket_bytes_cap,
                    gradient_scaling_factor=gradient_scaling_factor,
                    ddp_handle=self,
                )
            )
            for param in params:
                self.param_to_buffer[param] = buffers[-1]
        logger.debug("allocate buffers for parameters:", buffers)
        return buffers

    def final_grad_reduce(self):
        for buffer in self.buffers:
            buffer.final_grad_reduce()
            buffer.issued = 0
        self.buffer_issued = 0

    def _register_hook_for_params(self):
        """register backward hook for each params."""
        for param in self.module.get_parameters():
            if param.requires_grad:
                param.register_hook(self._make_param_hook(param))

    def _post_forward(self):
        pass

    def _pre_forward(self):
        """pre-process of forward pass to allocate buffer for parameters."""
        if self.rebuilt_params_ and self._should_rebuild_buckets():
            for i in self.rebuilt_params_:
                i.old_grad = i.grad

            self.buffers = self.allocate_buffers_for_parameters(
                self.rebuilt_params_,
                group=get_data_parallel_group()
                if self.process_group is None
                else self.process_group,
                gradient_scaling_factor=self.gradient_scaling_factor,
            )
            for buffer in self.buffers:
                buffer.sync_enabled = self.sync_enabled

            for i in self.rebuilt_params_:
                i.grad.copy_(i.old_grad)
                i.old_grad = None

            self.has_bucket_rebuilt = True
            self.rebuilt_params_ = []

    def construct(self, *inputs, **inputs_dict):
        """construct for DistributedDataParallel."""
        self._pre_forward()
        output = self.module(*inputs, **inputs_dict)
        self._post_forward()
        return output

    def zero_grad(self):
        """reset buffers for the next train iteration."""
        for buffer in self.buffers:
            buffer.reset()

    def _enable_sync(self, enable):
        """enable grad buffer sync or not."""
        for buffer in self.buffers:
            buffer.sync_enabled = enable
        self.sync_enabled = enable

    @contextmanager
    def no_sync(self):
        """context manager helper function."""
        self._enable_sync(False)
        try:
            yield
        finally:
            self._enable_sync(True)

    def _should_rebuild_buckets(self):
        if self.static_graph and not self.has_bucket_rebuilt:
            return True
        return False

    def _make_param_hook(self, param):
        """make closure function as the param hook."""
        def param_hook(grad):
            buffer = self.param_to_buffer[param]
            param.grad.add_(grad)
            buffer.register_grad_ready(param)
            if self._should_rebuild_buckets():
                self.rebuilt_params_.append(param)
            return param.grad

        return param_hook
