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
"""pipeline schedule"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from ._utils import _MicroBatch


class MetaStepType(Enum):
    """Specify the enumeration type for MetaStep."""
    FWD = auto()
    BWD = auto()
    FWD_RECV = auto()
    FWD_SEND = auto()
    BWD_RECV = auto()
    BWD_SEND = auto()


class MetaStep:
    """
    Meta step of PipelineSchedule.
    An execution list composed of MetaStep can be constructed
    and fed into the PipelineSchedule for execution.

    Args:
        micro_index (int): The index of micro-batch.
        type (MetaStepType): Specify the type of current step.
        stage_index(int): Specify the stage index of current step.
    """
    def __init__(self, micro_index, type, stage_index):
        self._type = type
        self._micro_index = micro_index
        self._stage_index = stage_index

    @property
    def micro_index(self):
        return self._micro_index

    @property
    def stage_index(self):
        return self._stage_index

    @property
    def type(self):
        return self._type

    def __str__(self):
        return f"MetaStep(type={self.type}, micro_index={self.micro_index}, stage_index={self.stage_index})"

    def __repr__(self):
        return f"MetaStep(type={self.type}, micro_index={self.micro_index}, stage_index={self.stage_index})"

    @staticmethod
    def from_str(step_str):
        pass


class PipelineScheduleBase(ABC):
    """
    Base class for pipeline schedule.
    Implements the `split_microbatches` method.
    Derived classes should implement `run_microbatches` method and `run` method.

    Args:
        micro_batch_num (int): The number of micro-batch.
        args_batch_dim (list, optional): Specify the batch dim of the args.
            Default ``None``.
        kwargs_batch_dim(dict, optional): Specify the batch dim of the kwargs.
            Default ``None``.
    """
    def __init__(self, micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None,
                 output_concat_dim=None, scale_grads=True):
        self.micro_batch_num = micro_batch_num
        self._args_batch_dim = args_batch_dim
        self._kwargs_batch_dim = kwargs_batch_dim
        self._output_concat_dim = output_concat_dim
        self._scale_grads = scale_grads
        self.split_micro_batch = _MicroBatch(self.micro_batch_num, self._args_batch_dim, self._kwargs_batch_dim)

    def split_microbatches(self, args, kwargs):
        if args or kwargs:
            args_split, kwargs_split = self.split_micro_batch(args, kwargs)
            return args_split, kwargs_split
        return [[]] * self.micro_batch_num, [{}] * self.micro_batch_num

    @abstractmethod
    def run_microbatches(self, arg_mbs, kwarg_mbs):
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class PipelineScheduleSingle(PipelineScheduleBase):
    """
    Base class for pipeline schedule with single-stage.
    Implements the `run` and `run_microbatches` method.
    Derived classes should implement `_construct_exec_order`.
    Args:
        stage (PipelineStage): A Pipeline stage representing partial of Model.
        micro_batch_num (int): The number of micro-batch.
        args_batch_dim (list, optional): Specify the batch dim of the args.
            Default ``None``.
        kwargs_batch_dim(dict, optional): Specify the batch dim of the kwargs.
            Default ``None``.
        scale_grads(bool): Whether to scale grads by a factor of 1/micro_batches.
    """
    def __init__(self,
                 stage,
                 micro_batch_num,
                 args_batch_dim=None,
                 kwargs_batch_dim=None,
                 output_concat_dim=None,
                 scale_grads=True):
        super().__init__(micro_batch_num,
                         args_batch_dim=args_batch_dim,
                         kwargs_batch_dim=kwargs_batch_dim,
                         output_concat_dim=output_concat_dim,
                         scale_grads=scale_grads)
        self.stage = stage
        self.exec_order = {}
        self.construct_exec_order()

    @abstractmethod
    def construct_exec_order(self):
        raise NotImplementedError

    def run_microbatches(self, arg_mbs, kwarg_mbs):
        out_list = []
        grad_out = None
        for cur_step in self.exec_order[self.stage.stage_index]:
            micro_index = cur_step.micro_index
            if cur_step.type == MetaStepType.FWD_RECV:
                self.stage.exec_fwd_recv_ops(micro_index)
            if cur_step.type == MetaStepType.FWD:
                out = self.stage.forward_one_chunk(micro_index, arg_mbs[micro_index], kwarg_mbs[micro_index])
                out_list.append(out)
            if cur_step.type == MetaStepType.FWD_SEND:
                self.stage.exec_fwd_send_ops(micro_index)
            if cur_step.type == MetaStepType.BWD_RECV:
                self.stage.exec_bwd_recv_ops(micro_index)
            if cur_step.type == MetaStepType.BWD:
                if micro_index == self.micro_batch_num - 1:
                    grad_out = self.stage.backward_one_chunk(micro_index, True)
                else:
                    _ = self.stage.backward_one_chunk(micro_index)
            if cur_step.type == MetaStepType.BWD_SEND:
                self.stage.exec_bwd_send_ops(micro_index)
        self.stage.sync_shared_parameters_grad()
        return out_list, grad_out

    def run(self, *args, **kwargs):
        split_args, split_kwargs = self.split_microbatches(args, kwargs)
        out = self.run_microbatches(split_args, split_kwargs)
        return out


class ScheduleGPipe(PipelineScheduleSingle):
    """
    The Gpipe schedule.
    It first executes all forward micro batches and then execute all backward micro batches.
    """
    def construct_exec_order(self):
        for stage_index in range(self.stage.stage_num):
            order_list = []
            for mb_index in range(self.micro_batch_num):
                if stage_index != 0:
                    order_list.append(MetaStep(mb_index, MetaStepType.FWD_RECV, stage_index))
                order_list.append(MetaStep(mb_index, MetaStepType.FWD, stage_index))
                if stage_index != self.stage.stage_num - 1:
                    order_list.append(MetaStep(mb_index, MetaStepType.FWD_SEND, stage_index))
            for mb_index in range(self.micro_batch_num):
                if stage_index != self.stage.stage_num - 1:
                    order_list.append(MetaStep(mb_index, MetaStepType.BWD_RECV, stage_index))
                order_list.append(MetaStep(mb_index, MetaStepType.BWD, stage_index))
                if stage_index != 0:
                    order_list.append(MetaStep(mb_index, MetaStepType.BWD_SEND, stage_index))
            self.exec_order[stage_index] = order_list


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    It will perform one forward and one backward on the micro batches in steady state.
    """
    def construct_exec_order(self):
        for stage_index in range(self.stage.stage_num):
            order_list = []
            fwd_index = 0
            bwd_index = 0
            # warmup phase
            warmup_micro_batches = min(self.stage.stage_num - stage_index, self.micro_batch_num)
            for _ in range(warmup_micro_batches):
                if stage_index != 0:
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD_RECV, stage_index))
                if stage_index % 2 == 0:
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD, stage_index))
                    if fwd_index != warmup_micro_batches - 1:
                        order_list.append(MetaStep(fwd_index, MetaStepType.FWD_SEND, stage_index))
                else:
                    if fwd_index > 0:
                        order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD, stage_index))
                fwd_index += 1

            # if warmup phase cannot filled up, then we need to execute fwd send in advance
            if self.stage.stage_num - stage_index > self.micro_batch_num:
                order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                fwd_index += 1

            # steady phase
            steady_micro_batches = self.micro_batch_num - warmup_micro_batches
            for _ in range(steady_micro_batches):
                if stage_index != self.stage.stage_num - 1:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_RECV, stage_index))
                    order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                order_list.append(MetaStep(bwd_index, MetaStepType.BWD, stage_index))

                if stage_index != 0:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_SEND, stage_index))
                    order_list.append(MetaStep(fwd_index, MetaStepType.FWD_RECV, stage_index))
                order_list.append(MetaStep(fwd_index, MetaStepType.FWD, stage_index))
                fwd_index += 1
                bwd_index += 1

            # cooldown phase
            cooldown_micro_batches = warmup_micro_batches
            for _ in range(cooldown_micro_batches):
                if stage_index != self.stage.stage_num - 1:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_RECV, stage_index))
                    if bwd_index == self.micro_batch_num - warmup_micro_batches and fwd_index <= self.micro_batch_num:
                        order_list.append(MetaStep(fwd_index - 1, MetaStepType.FWD_SEND, stage_index))
                order_list.append(MetaStep(bwd_index, MetaStepType.BWD, stage_index))

                if stage_index != 0:
                    order_list.append(MetaStep(bwd_index, MetaStepType.BWD_SEND, stage_index))
                bwd_index += 1
            self.exec_order[stage_index] = order_list
