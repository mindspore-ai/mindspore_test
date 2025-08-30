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
"""pipeline stage"""
from abc import ABC
import mindspore as ms
from mindspore import ops, Tensor, nn, mint
from mindspore.mint.distributed import isend, irecv, get_global_rank
from ._utils import _RecvInfo


class P2PInfo:
    """
    Used for inputing P2P communication information, including
    shape, dtype, rank and param_name.

    Args:
        dtype (dtype): The dtype of p2p input tensor.
        shape (list): The shape of p2p input tensor. In dynamic shape scenarios, the dynamic dim shuold be set to -1.
        src_stage(int): The source stage of receive op. Default ``None``.
        dst_stage(int): The destination stage of send op. Default ``None``.
        dyn_shape(bool): Specify whether the P2P operator has a dynamic shape. Default ``False``.
        dyn_rank(bool): Specify whether the P2P operator has a dynamic rank. Default ``False``.
    """
    def __init__(self, dtype, shape, src_stage=None, dst_stage=None, dyn_shape=False, dyn_rank=False):
        self._shape = shape
        self._dtype = dtype
        self._src_stage = src_stage
        self._dst_stage = dst_stage
        self._dyn_shape = dyn_shape
        self._dyn_rank = dyn_rank
        if self._dyn_rank and not self._dyn_shape:
            self._dyn_shape = True

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def src_stage(self):
        return self._src_stage

    @property
    def dst_stage(self):
        return self._dst_stage

    @property
    def dyn_shape(self):
        return self._dyn_shape

    @property
    def dyn_rank(self):
        return self._dyn_rank


class PipelineStage(ABC):
    """
    PipelineStage represents a pipeline stage in pipeline parallelism.

    PipelineStage requires the input of a segmented model.

    PipelineStage encapsulates the forward and backward functions used in PipelineSchedule,
    as well as P2P communication.

    Args:
        submodule (nn.Cell): Segmented model.
        stage_index (int): Stage index of current stage.
        stage_num (int): Total stage number.
        group (str): Group of p2p communication.
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        recv_info(P2PInfo, optional): Specify Receive information. Default ``None``.
        send_info(P2PInfo, optional): Specify Send information. Default ``None``.
    """
    def __init__(self, submodule: nn.Cell, stage_index: int, stage_num: int, group: str,
                 has_backward=True, recv_info=None, send_info=None):
        super().__init__()
        self.submodule = submodule
        self.stage_index = stage_index
        self.stage_num = stage_num
        self.pp_group = group
        self._has_backward = has_backward
        self._recv_info = self._check_p2p_info(recv_info)
        self._send_info = self._check_p2p_info(send_info)
        self._recv_num = len(recv_info)
        self._backward_func = None
        self._construct_backward_func()
        self.fwd_inputs_cache = {}
        self.fwd_outputs_cache = {}
        self.args_recv_info = {}
        self.grad_recv_info = {}
        self.bwd_cache = {}

    def clear_cache(self):
        self.fwd_inputs_cache.clear()
        self.fwd_outputs_cache.clear()
        self.bwd_cache.clear()

    def init_states(self, microbatches_num):
        self._prepare_forward_infra(microbatches_num)
        self._prepare_backward_infra(microbatches_num)

    def clear_states(self):
        self.args_recv_info.clear()
        self.grad_recv_info.clear()

    def _check_p2p_info(self, p2p_info):
        """check type for send_info and recv_info"""
        if p2p_info is None:
            return p2p_info

        if isinstance(p2p_info, P2PInfo):
            p2p_info_list = [p2p_info]
            return p2p_info_list

        if isinstance(p2p_info, (list, tuple)):
            for each_info in p2p_info:
                if not isinstance(each_info, P2PInfo):
                    raise TypeError(f"Argument send_info and recv_info must be of type None, P2PInfo, \
                                     list/tuple of P2PInfo, but got type list/tuple of {each_info}.")
            return p2p_info

        raise TypeError(f"Argument send_info and recv_info must be of type None, P2PInfo, \
                          list/tuple of P2PInfo, but got {p2p_info}.")

    # TODO: To adapt to dynamic shape, the shape in recv_info is set to -1. The shape iis transmitted
    #       through an additional pair of send and recv operations.
    def _make_tensor(self, recv_info, global_rank):
        """create recv buffer."""
        shape_dim = None
        if recv_info.dyn_rank:
            shape_dim = self._communicate_rank(global_rank)
        else:
            shape_dim = len(recv_info.shape)
        shape = None
        if recv_info.dyn_shape:
            shape = self._communicate_shape(global_rank, shape_dim)
        else:
            shape = recv_info.shape
        return mint.empty(shape, dtype=recv_info.dtype)

    def _communicate_shape(self, global_rank, shape_dim=None, tensor_send=None):
        if tensor_send is not None:
            handle = isend(Tensor(tensor_send.shape, dtype=ms.int64), global_rank)
            handle.wait()
            return None

        recv_tensor = mint.empty([shape_dim], dtype=ms.int64)
        handle = irecv(recv_tensor, global_rank)
        handle.wait()
        return recv_tensor.tolist()

    def _communicate_rank(self, global_rank, tensor_send=None):
        if tensor_send is not None:
            handle = isend(Tensor([tensor_send.ndim], dtype=ms.int64), global_rank)
            handle.wait()
            return None

        recv_tensor = mint.empty([1], dtype=ms.int64)
        handle = irecv(recv_tensor, global_rank)
        handle.wait()
        return recv_tensor.tolist()[0]

    def _init_recv_buffer(self, recv_info, global_rank):
        recv_info.buffer = self._make_tensor(recv_info, global_rank)

    def _clear_recv_buffer(self, micro_index):
        if micro_index not in self.args_recv_info.keys():
            return
        for info in self.args_recv_info[micro_index]:
            info.buffer = None
        if micro_index not in self.grad_recv_info.keys():
            return
        for info in self.grad_recv_info[micro_index]:
            info.buffer = None

    @property
    def is_first_stage(self):
        return self.stage_index == 0

    @property
    def is_last_stage(self):
        return self.stage_index == self.stage_num - 1

    def _prepare_forward_infra(self, microbatches_num):
        """_prepare_forward_infra"""
        for mbs_index in range(microbatches_num):
            recv_infos = []
            for info in self._recv_info:
                recv_info = _RecvInfo.from_instance(info)
                src_stage = self.stage_index - 1 if recv_info.src_stage is None else recv_info.src_stage
                recv_info.src_stage = src_stage
                recv_infos.append(recv_info)
            self.args_recv_info[mbs_index] = recv_infos

    def _prepare_backward_infra(self, microbatches_num):
        """_prepare_backward_infra"""
        for mbs_index in range(microbatches_num):
            recv_infos = []
            for info in self._send_info:
                recv_info = _RecvInfo.from_instance(info)
                recv_info.src_stage = self.stage_index + 1 if info.dst_stage is None else info.dst_stage
                recv_infos.append(recv_info)
            self.grad_recv_info[mbs_index] = recv_infos

    def forward_one_chunk(self, micro_index, args=None, kwargs=None):
        """Execution a forward function"""
        composite_args = args or []
        composite_kwargs = kwargs or {}
        recv_args = []
        if micro_index in self.args_recv_info.keys():
            recv_args = [recv_info.buffer for recv_info in self.args_recv_info[micro_index]]
        composite_args.extend(recv_args)
        out = self.submodule(*composite_args, **composite_kwargs)
        out_tuple = out if isinstance(out, tuple) else (out,)
        self.fwd_inputs_cache[micro_index] = (composite_args, composite_kwargs)
        self.fwd_outputs_cache[micro_index] = out_tuple
        return out

    def backward_one_chunk(self, micro_index):
        """Execution a backward function"""
        fwd_args, fwd_kwargs = self.fwd_inputs_cache.pop(micro_index)
        recv_args = []
        if micro_index in self.grad_recv_info.keys():
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]
        grad_out = None
        if self.is_first_stage:
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=recv_args)
        elif self.is_last_stage:
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs)
        else:
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=recv_args)
        self._clear_recv_buffer(micro_index)
        if not self.is_first_stage:
            self.bwd_cache[micro_index] = grad_out[0][-self._recv_num :]
        return grad_out

    def exec_fwd_recv_ops(self, micro_index):
        """Execute the forward recv operation"""
        if micro_index not in self.args_recv_info.keys():
            return
        for recv_info in self.args_recv_info[micro_index]:
            global_rank = get_global_rank(self.pp_group, recv_info.src_stage)
            self._init_recv_buffer(recv_info, global_rank)
            handle = irecv(recv_info.buffer, global_rank)
            handle.wait()

    def exec_fwd_send_ops(self, micro_index):
        """Execute the forward send operation"""
        if not self._send_info:
            return
        out = self.fwd_outputs_cache.pop(micro_index)
        for idx, send_info in enumerate(self._send_info):
            dst_stage = self.stage_index + 1 if send_info.dst_stage is None else send_info.dst_stage
            global_rank = get_global_rank(self.pp_group, dst_stage)
            if send_info.dyn_rank:
                self._communicate_rank(global_rank, tensor_send=out[idx])
            if send_info.dyn_shape:
                self._communicate_shape(global_rank, tensor_send=out[idx])
            handle = isend(out[idx], global_rank)
            handle.wait()

    def exec_bwd_recv_ops(self, micro_index):
        """Execute the backward recv operation"""
        if micro_index not in self.grad_recv_info.keys():
            return
        for recv_info in self.grad_recv_info[micro_index]:
            src_stage = self.stage_index + 1 if recv_info.src_stage is None else recv_info.src_stage
            global_rank = get_global_rank(self.pp_group, src_stage)
            self._init_recv_buffer(recv_info, global_rank)
            handle = irecv(recv_info.buffer, global_rank)
            handle.wait()

    def exec_bwd_send_ops(self, micro_index):
        """Execute the backward send operation"""
        if micro_index not in self.args_recv_info.keys():
            return
        out = self.bwd_cache.pop(micro_index)
        for idx, info in enumerate(self.args_recv_info[micro_index]):
            global_rank = get_global_rank(self.pp_group, info.src_stage)
            if info.dyn_rank:
                self._communicate_rank(global_rank, tensor_send=out[idx])
            if info.dyn_shape:
                self._communicate_shape(global_rank, tensor_send=out[idx])
            handle = isend(out[idx], global_rank)
            handle.wait()

    def _construct_backward_func(self):
        self._backward_func = None
        if self.is_first_stage:
            self._backward_func = ops.GradOperation(get_by_list=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())
        elif self.is_last_stage:
            self._backward_func = ops.GradOperation(get_by_list=True, get_all=True, sens_param=False)(
                self.submodule, self.submodule.trainable_params())
        else:
            self._backward_func = ops.GradOperation(get_by_list=True, get_all=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())
