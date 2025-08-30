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
"""pipeline parallel utils"""
from mindspore import nn
import mindspore.ops as ops


class BatchDimSpec:
    """
    Specify the batch dimension of a Tensor.

    Args:
        batch_dim(int): batch dimension。
    """
    __slots__ = ("batch_dim",)

    def __init__(self, batch_dim):
        if not isinstance(batch_dim, int):
            raise TypeError(f"batch_dim must be int.")
        self.batch_dim = batch_dim

    def __repr__(self):
        return f"BatchDimSpec({self.batch_dim})"

    def __str__(self):
        return f"BatchDim(dim={self.batch_dim})"

    @staticmethod
    def from_tuple(batch_dims):
        if not isinstance(batch_dims, tuple):
            raise TypeError(f"batch_dims must be tuple.")
        return tuple(BatchDimSpec(dim) for dim in batch_dims)

    @staticmethod
    def from_dict(batch_dims):
        if not isinstance(batch_dims, dict):
            raise TypeError(f"batch_dims must be dict.")
        return {k: BatchDimSpec(v) for k, v in batch_dims.items()}


class _MicroBatch(nn.Cell):
    """
    Split inputs into micro_batch in pipeline parallel.

    Args:
        micro_batch_num (int): The number of micro-batch.
        args_batch_dim (list, optional): Specify the batch dim of the args.
            Default ``None``.
        kwargs_batch_dim(dict, optional): Specify the batch dim of the kwargs.
            Default ``None``.
    Inputs:
        - **args** (list) - Input args.
        - **kwargs** (dict) - Input kwargs.

    Outputs:
        - **args_after_split** (list) - Input args after split into micro_batches.
        - **kwargs_after_split** (list) - Input kwargs after split into micro_batches.
    """
    def __init__(self, micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        super().__init__()
        self.micro_batch_num = micro_batch_num
        self.args_batch_dim = args_batch_dim
        self.kwargs_batch_dim = kwargs_batch_dim

    def construct(self, args, kwargs):
        """Construct of _MicroBatch"""
        args_after_split = []
        kwargs_after_split = []
        for micro_idx in range(self.micro_batch_num):
            micro_args = []
            micro_kwargs = {}
            for arg_idx, cur_arg in enumerate(args):
                cur_arg_batch_dim = 0
                if self.args_batch_dim and self.args_batch_dim[arg_idx] is not None:
                    cur_arg_batch_dim = self.args_batch_dim[arg_idx]
                micro_arg = self.split_inputs(cur_arg, cur_arg_batch_dim, micro_idx)
                micro_args.append(micro_arg)
            args_after_split.append(micro_args)

            for key, cur_kwarg in kwargs:
                cur_kwarg_batch_dim = 0
                if self.kwargs_batch_dim is not None:
                    cur_kwarg_batch_dim = self.kwargs_batch_dim[key]
                micro_kwarg = self.split_inputs(cur_kwarg, cur_kwarg_batch_dim, micro_idx)
                micro_kwargs[key] = micro_kwarg
            kwargs_after_split.append(micro_kwargs)
        return args_after_split, kwargs_after_split

    def split_inputs(self, input, cur_arg_batch_dim, micro_idx):
        """
        Split the input along the specified batch_dim and micro_idx
        """
        if cur_arg_batch_dim == -1:
            return input
        batch_dim_shape = input.shape[cur_arg_batch_dim]
        micro_batch_begin = (batch_dim_shape // self.micro_batch_num) * micro_idx
        micro_batch_end = (batch_dim_shape // self.micro_batch_num) * (micro_idx + 1)
        strided_slice_begin = [0] * input.ndim
        strided_slice_strides = [1] * input.ndim
        strided_slice_end = list(input.shape)
        strided_slice_begin[cur_arg_batch_dim] = micro_batch_begin
        strided_slice_end[cur_arg_batch_dim] = micro_batch_end
        micro_input = ops.strided_slice(input, strided_slice_begin, strided_slice_end, strided_slice_strides)
        return micro_input


class _RecvInfo:
    """
    Used for construct forward Receive operation and backward Send operation.
    """
    def __init__(self, dtype, shape, src_stage, dyn_shape, dyn_rank):
        self._src_stage = src_stage
        self.buffer = None
        self._shape = shape
        self._dtype = dtype
        self._dyn_shape = dyn_shape
        self._dyn_rank = dyn_rank
        self.src_stage = src_stage

    @classmethod
    def from_instance(cls, recv_info):
        return cls(recv_info.dtype, recv_info.shape, recv_info.src_stage,
                   recv_info.dyn_shape, recv_info.dyn_rank)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def dyn_shape(self):
        return self._dyn_shape

    @property
    def dyn_rank(self):
        return self._dyn_rank
