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
