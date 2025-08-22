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
"""HSDP parameter"""
import functools
from mindspore import ops, Parameter, Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore.common.initializer import initializer
from mindspore.common.dtype import type_size_in_bytes

class HSDPParam:
    """
    HSDP parameter.
    """
    def __init__(self, cell, param_name, param, hsdp_state):
        self.cell = cell
        self.param_name = param_name
        self.param = param
        self.requires_acc_grad = hsdp_state.requires_acc_grad
        self.shard_size = hsdp_state.shard_size
        self.threshold = hsdp_state.threshold
        self.is_shard_level1 = hsdp_state.is_shard_level1
        self.unsharded_param = None
        self.sharded_param = None
        self.acc_grad = None
        self.not_sharded = False
        self.fully_sharded = True
        self.comm = hsdp_state.comm
        self.rank_id = get_rank()
        self.rank_size = get_group_size()
        self._init()
        self.dp_size = self.rank_size // self.shard_size
        self.dp_mean_factor = 1.0 / self.dp_size
        self.op_mean_factor = 1.0 / self.shard_size
        self.sharded_group_name = self._create_sharded_dp_group()
        self.unsharded_group_name = self._create_unsharded_dp_group()

    def _create_sharded_dp_group(self):
        """create communication group for sharded parameter"""
        if self.shard_size <= 1:
            return "hsdp_sharded_dp_group_invalid"
        rank_base = self.rank_id // self.shard_size * self.shard_size
        rank_list = [i + rank_base for i in range(self.shard_size)]
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_sharded_dp_group_" + rank_list_str
        self.comm.create_group(group_name, rank_list)
        return group_name

    def _create_unsharded_dp_group(self):
        """create communication group for unsharded parameter"""
        if self.dp_size <= 1:
            return "hsdp_unsharded_dp_group_invalid"
        rank_stride = self.shard_size
        rank_base = self.rank_id % rank_stride
        rank_list = [i * rank_stride + rank_base for i in range(self.dp_size)]
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_unshared_dp_group_" + rank_list_str
        self.comm.create_group(group_name, rank_list)
        return group_name

    def _init(self):
        """init hsdp parameter"""
        self.param.acc_grad = None
        param_size = functools.reduce(lambda x, y: x * y, self.param.shape, type_size_in_bytes(self.param.dtype))
        if (self.shard_size == 1 or
                param_size < self.threshold or
                self.param.shape[0] < self.shard_size or
                self.param.shape[0] % self.shard_size != 0):
            self.shard_size = 1
            self.sharded = False
            self.fully_sharded = False
            self.param.init_data()
            if self.requires_acc_grad:
                self.acc_grad = Parameter(initializer("zeros", self.param.shape, self.param.dtype),
                                          name="acc_grad_"+self.param.name,
                                          requires_grad=False)
            self.param.acc_grad = self.acc_grad
            return

        origin_param_shape = [i for i in self.param.shape]
        if not self.param.has_init:
            slice_index = self.rank_id % self.shard_size
            param_slice = ops.split(self.param, self.param.shape[0] // self.shard_size)[slice_index]
            self.sharded_param = Parameter(Tensor(param_slice, self.param.dtype),
                                           name="sharded_"+self.param.name,
                                           requires_grad=False)
        else:
            init_shape = [i for i in self.param.init_mode.shape]
            init_shape[0] = init_shape[0] // self.shard_size
            self.param.init_mode.shape = init_shape
            self.param.init_data()
            self.sharded_param = Parameter(Tensor(self.param.numpy(), self.param.dtype),
                                           name="sharded_"+self.param.name,
                                           requires_grad=False)
        if self.requires_acc_grad:
            acc_grad_shape = origin_param_shape
            if not self.is_shard_level1:
                acc_grad_shape = self.sharded_param.shape
            self.acc_grad = Parameter(initializer("zeros", acc_grad_shape, self.param.dtype),
                                      name="acc_grad_"+self.param.name,
                                      requires_grad=False)
        self.param.acc_grad = self.acc_grad
        self.to_sharded()
        self.sharded = True
        if self.shard_size == self.rank_size:
            self.fully_sharded = True
        else:
            self.fully_sharded = False

    def to_sharded(self):
        """change parameter to sharded state"""
        self.param.set_data(self.sharded_param)

    def to_unsharded(self):
        """change parameter to unsharded state"""
        unshared_param_data = self.comm.all_gather(self.sharded_group_name, self.param)
        self.sharded_param.set_data(self.param)
        self.param.set_data(unshared_param_data)

    def zero_acc_grad(self):
        """zero accumunication grad"""
        if self.param.acc_grad is not None:
            self.param.acc_grad.zero_()
