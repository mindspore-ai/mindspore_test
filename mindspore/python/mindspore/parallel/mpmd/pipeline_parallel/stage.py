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
from mindspore import ops, Tensor, nn, mint
from mindspore.mint.distributed import isend, irecv, get_global_rank


class P2PInfo:
    """
    Used for inputting P2P communication information, including
    shape, dtype, rank and param_name.

    Args:
        dtype (dtype): The dtype of p2p input tensor.
        shape (list): The shape of p2p input tensor.
        relative_rank(int): The source rank or target rank of p2p op.
        param_name(str): The formal parameter name corresponding to Receive operation.
    """
    def __init__(self, dtype, shape, relative_rank=None, param_name=None):
        self._shape = shape
        self._dtype = dtype
        self._relative_rank = relative_rank
        self._param_name = param_name

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def relative_rank(self):
        return self._relative_rank

    @property
    def param_name(self):
        return self._param_name


class _RecvInfo(ABC):
    """
    Used for construct forward Receive operation and backward Send operation.
    """
    def __init__(self, dtype, shape, source_stage: int, buffer: Tensor = None):
        self._source_stage = source_stage
        self.buffer = buffer
        self._shape = shape
        self._dtype = dtype

    @property
    def source_stage(self):
        return self._source_stage

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype


class _SendInfo(ABC):
    """
    Used for construct forward Send operation and backward Receive operation.
    """
    def __init__(self, target_stage: int):
        self._target_stage = target_stage

    @property
    def target_stage(self):
        return self._target_stage


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
        self.args_send_info = {}
        self.grad_send_info = {}
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
        self.args_send_info.clear()
        self.grad_send_info.clear()

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
                    raise TypeError(f"Argument recv_info and send_info must be of type None, P2PInfo, \
                                     list/tuple of P2PInfo, but got type list/tuple of {each_info}.")
            return p2p_info

        raise TypeError(f"Argument recv_info and send_info must be of type None, P2PInfo, \
                          list/tuple of P2PInfo, but got type {p2p_info}.")

    # TODO: To adapt to dynamic shape, the shape in recv_info is set to -1. The shape iis transmitted
    #       through an additional pair of send and recv operations.
    def _make_tensor(self, recv_info):
        return mint.empty(recv_info.shape, dtype=recv_info.dtype)

    def _init_recv_buffer(self, micro_index):
        if micro_index not in self.args_recv_info.keys():
            return
        for info in self.args_recv_info[micro_index]:
            info.buffer = self._make_tensor(info)

    def _clear_recv_buffer(self, micro_index):
        if micro_index not in self.args_recv_info.keys():
            return
        for info in self.args_recv_info[micro_index]:
            info.buffer = None

    def _init_grad_recv_buffer(self, micro_index):
        if micro_index not in self.grad_recv_info.keys():
            return
        for info in self.grad_recv_info[micro_index]:
            info.buffer = self._make_tensor(info)

    def _clear_grad_recv_buffer(self, micro_index):
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
                source_stage = self.stage_index - 1 if info.relative_rank is None else info.relative_rank
                recv_info = _RecvInfo(info.dtype, info.shape, source_stage, None)
                recv_infos.append(recv_info)
            self.args_recv_info[mbs_index] = recv_infos

            send_infos = []
            for info in self._send_info:
                target_stage = self.stage_index + 1 if info.relative_rank is None else info.relative_rank
                send_info = _SendInfo(target_stage)
                send_infos.append(send_info)
            self.args_send_info[mbs_index] = send_infos

    def _prepare_backward_infra(self, microbatches_num):
        """_prepare_backward_infra"""
        for mbs_index in range(microbatches_num):
            recv_infos = []
            for info in self._send_info:
                source_stage = self.stage_index + 1 if info.relative_rank is None else info.relative_rank
                recv_info = _RecvInfo(info.dtype, info.shape, source_stage, None)
                recv_infos.append(recv_info)
            self.grad_recv_info[mbs_index] = recv_infos

            send_infos = []
            for info in self._recv_info:
                target_stage = self.stage_index - 1 if info.relative_rank is None else info.relative_rank
                send_info = _SendInfo(target_stage)
                send_infos.append(send_info)
            self.grad_send_info[mbs_index] = send_infos

    def forward_one_chunk(self, micro_index, args=None, kwargs=None):
        """Execution a forward function"""
        composite_args = args or []
        composite_kwargs = kwargs or {}
        recv_args = []
        if micro_index in self.args_recv_info.keys():
            recv_args = [recv_info.buffer for recv_info in self.args_recv_info[micro_index]]
        recv_args, recv_kwargs = self._map_argument(recv_args)
        composite_args.extend(recv_args)
        composite_kwargs.update(recv_kwargs)
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
        self._clear_grad_recv_buffer(micro_index)
        if not self.is_first_stage:
            self.bwd_cache[micro_index] = grad_out[0][-self._recv_num :]
        return grad_out

    def exec_fwd_recv_ops(self, micro_index):
        if micro_index not in self.args_recv_info.keys():
            return
        self._init_recv_buffer(micro_index)
        for recv_info in self.args_recv_info[micro_index]:
            global_rank = get_global_rank(self.pp_group, recv_info.source_stage)
            handle = irecv(recv_info.buffer, global_rank)
            handle.wait()

    def exec_fwd_send_ops(self, micro_index):
        if micro_index not in self.args_send_info.keys():
            return
        out = self.fwd_outputs_cache.pop(micro_index)
        for idx, send_info in enumerate(self.args_send_info[micro_index]):
            global_rank = get_global_rank(self.pp_group, send_info.target_stage)
            handle = isend(out[idx], global_rank)
            handle.wait()

    def exec_bwd_recv_ops(self, micro_index):
        if micro_index not in self.grad_recv_info.keys():
            return
        self._init_grad_recv_buffer(micro_index)
        for recv_info in self.grad_recv_info[micro_index]:
            global_rank = get_global_rank(self.pp_group, recv_info.source_stage)
            handle = irecv(recv_info.buffer, global_rank)
            handle.wait()

    def exec_bwd_send_ops(self, micro_index):
        if micro_index not in self.grad_send_info.keys():
            return
        out = self.bwd_cache.pop(micro_index)
        for idx, send_info in enumerate(self.grad_send_info[micro_index]):
            global_rank = get_global_rank(self.pp_group, send_info.target_stage)
            handle = isend(out[idx], global_rank)
            handle.wait()

    # For backward func, it is necessary to know the position information of the forward inputs.
    # Here, we use _map_argument to convert the recv_info's param_name to kwargs.
    # Currently, if recv_info's param_name is None, the default input order is (dataset_input, Recvs).
    def _map_argument(self, recv_outs):
        args = []
        kwargs = {}
        for idx, recv_out in enumerate(recv_outs):
            if self._recv_info[idx].param_name is not None:
                kwargs[param_name] = recv_out
            else:
                args.append(recv_out)
        return args, kwargs

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
