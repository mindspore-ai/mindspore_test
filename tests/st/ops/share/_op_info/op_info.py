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
"""Operator information schema and registry for tests.

Defines `OpInfo` dataclass and an in-repo registry for op metadata across
backends to drive parameterized tests.
"""
import torch
import mindspore as ms
from mindspore import mint
from typing import Callable, Optional, Dict
from dataclasses import dataclass, field
import functools
import inspect

dtypes_as_torch = (
    ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,
    ms.float16, ms.float32, ms.float64,
    ms.complex64, ms.complex128,
    ms.bfloat16,
)
dtypes_extra_uint = (
    ms.uint16, ms.uint32, ms.uint64,
)

dtypes_integral = (
    ms.bool_,
    ms.int8, ms.int16, ms.int32, ms.int64,
    ms.uint8, ms.uint16, ms.uint32, ms.uint64,
)

@dataclass
class OpInfo:
    """Metadata describing an operator under test.

    Attributes:
        name: Short op alias used in logs and test names.
        op: MindSpore callable implementation.
        ref: Reference implementation (e.g., torch, numpy).
        tensor_variant: Tensor method variant if applicable.
        inplace_variant: In-place variant if applicable.
        dtypes_ascend/cpu/gpu: Supported dtypes on each backend.
        dtypes_intersection: Intersection of supported dtypes across backends.
        is_differentiable: Whether gradients are expected/computed.
    """
    name: str
    op: Optional[Callable] = None
    ref: Optional[Callable] = None
    tensor_variant: Optional[Callable] = None
    inplace_variant: Optional[Callable] = None
    dtypes_ascend: tuple = field(default_factory=tuple)
    dtypes_cpu: tuple = field(default_factory=tuple)
    dtypes_gpu: tuple = field(default_factory=tuple)
    dtypes_intersection: tuple = field(default_factory=tuple)
    is_differentiable: bool = True

    def __post_init__(self):
        if not self.dtypes_intersection:
            self.dtypes_intersection = tuple(
                set(self.dtypes_ascend) & set(self.dtypes_cpu) & set(self.dtypes_gpu)
            )

    def get_dtypes(self, backend: str = None):
        if backend is None:
            return self.dtypes_intersection
        if backend.lower() == 'ascend':
            return self.dtypes_ascend
        if backend.lower() == 'cpu':
            return self.dtypes_cpu
        if backend.lower() == 'gpu':
            return self.dtypes_gpu
        raise ValueError(f"Invalid backend: {backend}, expected: 'ascend', 'cpu', 'gpu'.")


op_db: Dict[str, OpInfo] = {
    'add_ext': OpInfo(
        name='add_ext',
        op=mint.add,
        ref=torch.add,
        tensor_variant=lambda op_input, *op_args, **op_kwargs: op_input.add(op_args[0], alpha=op_kwargs.get('alpha', 1)),
        dtypes_ascend=dtypes_as_torch,
        dtypes_cpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16] + list(dtypes_extra_uint)),
        dtypes_gpu=tuple([d for d in dtypes_as_torch if d != ms.bfloat16] + list(dtypes_extra_uint)),
        is_differentiable=True,
    ),
}


def ops_info(op_info: OpInfo):
    """Decorator factory: can be used as @ops_info(op_db['add_ext']).

    Purpose:
    - Injects the provided op_info into the decorated function as the last argument.
    - Adjusts the exported signature to hide the `op_info` parameter for pytest compatibility
      (works with parametrization and stacked decorators).
    - Supports being stacked before or after pytest decorators.
    """

    def _decorator(fn: Callable):
        # Record the original signature and try to hide the trailing `op_info` parameter from export
        try:
            original_sig = inspect.signature(fn)
            params = list(original_sig.parameters.values())
        except (TypeError, ValueError):
            original_sig = None
            params = []
        has_opinfo_param = any(p.name == 'op_info' for p in params)

        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            # If caller explicitly provides `op_info` as a keyword, do not override it
            if 'op_info' in kwargs:
                return fn(*args, **kwargs)
            # If function declares `op_info`, pass it as a keyword to avoid conflicts with keyword args
            if has_opinfo_param:
                new_kwargs = dict(kwargs)
                new_kwargs['op_info'] = op_info
                return fn(*args, **new_kwargs)
            # Otherwise, append `op_info` as the last positional argument
            new_args = args + (op_info,)
            return fn(*new_args, **kwargs)

        # Expose a signature without the trailing `op_info` so pytest won't treat it as a fixture/param
        if has_opinfo_param and params and params[-1].name == 'op_info':
            try:
                _wrapper.__signature__ = original_sig.replace(parameters=tuple(params[:-1]))
            except Exception:  # pylint: disable=W0703
                pass
        return _wrapper

    return _decorator
