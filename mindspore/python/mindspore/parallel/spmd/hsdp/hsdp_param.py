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
import mindspore.ops as ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank, create_group, get_group_size
from mindspore.common.initializer import initializer
from mindspore.common.dtype import type_size_in_bytes
import mindspore.parallel.spmd.hsdp.hsdp_comm as comm
from mindspore.parallel.spmd.hsdp.hsdp_utils import OptimizerLevel


class HSDPParam:
    """
    HSDP parameter.
    """
    def __init__(self, cell, param_name, param, config):
        self.cell = cell
        self.param_name = param_name
        self.param = param
        self.config = config
        self.shard_size = 1
        self.unsharded_param = None
        self.sharded_param = None
        self.acc_grad = None
        self.sharded = False
        self.fully_sharded = True
        self._init_rank_info()
        self._init_param_shard_size()
        self._init_param()
        self.dp_size = self.rank_size // self.shard_size
        self.sharded_group_name = self._create_sharded_dp_group()
        self.unsharded_group_name = self._create_unsharded_dp_group()

    def _init_param_shard_size(self):
        """init parameter dp shard size"""
        if hasattr(self.param, "hsdp_shard_size"):
            if not isinstance(self.param.hsdp_shard_size, int) or \
                    (self.param.hsdp_shard_size <= 0 and self.param.hsdp_shard_size != -1):
                raise ValueError(f"param's hsdp_shard_size must be a positive integer, "
                                 f"but got {self.param.hsdp_shard_size}.")
            self.shard_size = self.param.hsdp_shard_size
        else:
            self.shard_size = self.config.shard_size

        param_size = functools.reduce(lambda x, y: x * y, self.param.local_shape, type_size_in_bytes(self.param.dtype))
        if param_size < self.config.threshold:
            self.shard_size = 1
            return
        if self.shard_size == -1 or self.param.local_shape[0] < self.shard_size:
            self.shard_size = self.param.local_shape[0]

        def _gcd(m, n):
            if m < n:
                m, n = n, m
            if n == 0:
                raise ValueError(f"HSDP invalid gcd input 0.")
            r = m % n
            if r == 0:
                return n
            return _gcd(n, r)

        rank_gcd = _gcd(self.param.local_shape[0], self.rank_size)
        if self.shard_size > rank_gcd:
            self.shard_size = rank_gcd
        if rank_gcd % self.shard_size != 0:
            self.shard_size = 1

    def _init_rank_info(self):
        """init parameter rank info"""
        self.rank_id = get_rank()
        self.hsdp_rank = self.rank_id
        self.local_rank = self.rank_id
        self.tp_rank = 0
        if self.param.layout is None:
            self.rank_size = get_group_size()
            return

        if len(self.param.layout.rank_list) == 1:
            self.rank_size = 1
            return

        try:
            self.local_rank = self.param.layout.rank_list.index(self.rank_id)
        except ValueError:
            raise ValueError(f"HSDP invalid rank {self.rank_id} with rank list {self.param.layout.rank_list}.")

        tensor_map = self.param.layout.tensor_map
        sharded_axis_set = set()
        for axis in tensor_map:
            if isinstance(axis, int) and axis != -1:
                sharded_axis_set.add(axis)
                continue
            if isinstance(axis, tuple):
                for item in axis:
                    sharded_axis_set.add(item)
        self.sharded_axis_set = sharded_axis_set
        self.rank_size = 1
        self.unsharded_reverse_axis_list = []
        self.global_rank_stride_list = []
        self.hsdp_rank_stride_list = []
        self.tp_rank_stride_list = []
        device_dims = len(self.param.layout.device_matrix)
        stride = 1
        hsdp_stride = 1
        tp_stride = 1
        for axis in range(device_dims):
            r_axis = device_dims - 1 - axis
            self.global_rank_stride_list.append(stride)
            self.hsdp_rank_stride_list.append(hsdp_stride)
            self.tp_rank_stride_list.append(tp_stride)
            stride = stride * self.param.layout.device_matrix[r_axis]
            if axis in self.sharded_axis_set:
                tp_stride = tp_stride * self.param.layout.device_matrix[r_axis]
                continue

            hsdp_stride = hsdp_stride * self.param.layout.device_matrix[r_axis]
            self.unsharded_reverse_axis_list.append(r_axis)
            self.rank_size = self.rank_size * self.param.layout.device_matrix[r_axis]
        self.global_rank_stride_list.reverse()
        self.hsdp_rank_stride_list.reverse()
        self.tp_rank_stride_list.reverse()

        rank_indices = []
        index = self.local_rank
        for stride in self.global_rank_stride_list:
            rank_indices.append(index // stride)
            index = index % stride
        self.rank_indices = rank_indices
        hsdp_rank = 0
        for axis in self.unsharded_reverse_axis_list:
            hsdp_rank = hsdp_rank + rank_indices[axis] * self.hsdp_rank_stride_list[axis]
        self.hsdp_rank = hsdp_rank
        tp_rank = 0
        for axis in range(device_dims):
            if axis in self.sharded_axis_set:
                r_axis = device_dims - 1 - axis
                tp_rank = tp_rank + rank_indices[r_axis] * self.tp_rank_stride_list[r_axis]
        self.tp_rank = tp_rank

    def _hsdp_rank_to_global_rank(self, hsdp_rank_list):
        """transform from hsdp rank to global rank"""
        rank_list = []
        for hsdp_rank in hsdp_rank_list:
            local_index = hsdp_rank
            local_indices_dict = {}
            for axis in self.unsharded_reverse_axis_list:
                stride = self.hsdp_rank_stride_list[axis]
                local_indices_dict[axis] = local_index // stride
                local_index = local_index % stride
            global_rank = 0
            for axis, index in enumerate(self.rank_indices):
                if axis in local_indices_dict:
                    index = local_indices_dict[axis]
                global_rank = global_rank + index * self.global_rank_stride_list[axis]
            if self.param.layout is not None:
                if global_rank >= len(self.param.layout.rank_list):
                    raise ValueError(f"HSDP invalid index {global_rank} with"
                                     f"rank list len {len(self.param.layout.rank_list)}.")
                global_rank = self.param.layout.rank_list[global_rank]
            rank_list.append(global_rank)
        return rank_list

    def _get_op_rank_list(self):
        """get data parallel rank list"""
        if self.param.layout is None:
            rank_base = self.local_rank // self.shard_size * self.shard_size
            rank_list = [i + rank_base for i in range(self.shard_size)]
            return rank_list

        rank_base = self.hsdp_rank // self.shard_size * self.shard_size
        hsdp_rank_list = [i + rank_base for i in range(self.shard_size)]
        return self._hsdp_rank_to_global_rank(hsdp_rank_list)

    def _get_dp_rank_list(self):
        """get optimizer parallel rank list"""
        if self.param.layout is None:
            rank_stride = self.shard_size
            rank_base = self.local_rank % rank_stride
            rank_list = [i * rank_stride + rank_base for i in range(self.dp_size)]
            return rank_list

        rank_stride = self.shard_size
        rank_base = self.hsdp_rank % rank_stride
        hsdp_rank_list = [i * rank_stride + rank_base for i in range(self.dp_size)]
        return self._hsdp_rank_to_global_rank(hsdp_rank_list)

    def _create_sharded_dp_group(self):
        """create communication group for sharded parameter"""
        if self.shard_size <= 1:
            return "hsdp_sharded_dp_group_invalid"

        rank_list = self._get_op_rank_list()
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_sharded_dp_group_" + rank_list_str
        create_group(group_name, rank_list)
        return group_name

    def _create_unsharded_dp_group(self):
        """create communication group for unsharded parameter"""
        if self.dp_size <= 1:
            return "hsdp_unsharded_dp_group_invalid"

        rank_list = self._get_dp_rank_list()
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_unshared_dp_group_" + rank_list_str
        create_group(group_name, rank_list)
        return group_name

    def _init_sharded_param(self):
        """add and init sharded param"""
        if not self.param.has_init:
            slice_index = self.hsdp_rank % self.shard_size
            if self.param.layout is None:
                param_slice = ops.split(self.param, self.param.local_shape[0] // self.shard_size)[slice_index]
            else:
                layout = self.param.layout
                self.param.to_local()
                param_slice = ops.split(self.param, self.param.local_shape[0] // self.shard_size)[slice_index]
                self.param.local_to_global(layout)
            self.sharded_param = Parameter(Tensor(param_slice, self.param.dtype),
                                           name="sharded_"+self.param.name,
                                           requires_grad=False)
        else:
            dp_slice_index = self.hsdp_rank % self.shard_size
            data_slice_index = self.tp_rank * self.shard_size + dp_slice_index
            init_shape = [i for i in self.param.init_mode.local_shape]
            init_shape[0] = init_shape[0] // self.shard_size
            init_data = self.param.init_mode.init_data(slice_index=data_slice_index, shape=init_shape)
            self.param.init_mode = None
            self.param.init = None
            self.param.set_data(init_data)
            self.sharded_param = Parameter(Tensor(self.param.numpy(), self.param.dtype),
                                           name="sharded_"+self.param.name,
                                           requires_grad=False)

    def _init_unsharded_param(self):
        """add and init unshared param when using param hook"""
        if self.config.use_cell_hook:
            return

        self.unsharded_param = Parameter(Tensor(self.param, self.param.dtype),
                                         name="unsharded_"+self.param.name,
                                         requires_grad=False)
        self.unsharded_param_available = Parameter(Tensor(False),
                                                   name="available_unsharded_"+self.param.name,
                                                   requires_grad=False)

    def _init_param(self):
        """init hsdp parameter"""
        self.param.acc_grad = None
        param_size = functools.reduce(lambda x, y: x * y, self.param.local_shape, type_size_in_bytes(self.param.dtype))
        if (self.shard_size == 1 or param_size < self.config.threshold):
            self.sharded = False
            self.fully_sharded = False
            self.param.init_data()
            if self.config.requires_acc_grad:
                self.acc_grad = Parameter(initializer("zeros", self.param.local_shape, self.param.dtype),
                                          name="acc_grad_"+self.param.name,
                                          requires_grad=False)
            self.param.acc_grad = self.acc_grad
            return

        origin_param_shape = [i for i in self.param.local_shape]
        self._init_unsharded_param()
        self._init_sharded_param()
        if self.config.requires_acc_grad:
            acc_grad_shape = origin_param_shape
            if self.config.shard_level != OptimizerLevel.SHARD_OPT:
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
        unshared_param_data, _ = comm.all_gather_into_tensor(self.param, group=self.sharded_group_name)
        self.sharded_param.set_data(self.param)
        self.param.set_data(unshared_param_data)

    def zero_acc_grad(self):
        """zero accumunication grad"""
        if self.param.acc_grad is not None:
            self.param.acc_grad.zero_()
