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
# pylint: disable=R1705
import math
import torch
import numpy as np
import mindspore as ms
from mindspore.common.api import _pynative_executor
from mindspore.common.dtype import _dtype_to_nptype
from typing import Optional

class OpSampleInput:

    __slots__ = [
        "op_input",
        "op_args",
        "op_kwargs",
        "op_name",
    ]

    def __init__(
            self,
            op_input,
            op_args: Optional[tuple] = tuple(),
            op_kwargs: Optional[dict] = None,
            op_name: Optional[str] = None,
    ):
        self.op_input = op_input
        self.op_args = op_args
        self.op_kwargs = op_kwargs if op_kwargs is not None else {}
        self.op_name = op_name if op_name is not None else "UnknownOp"

    def transform(self, fn, method):
        def _transform(x):
            def _transform_to(x):
                return fn(x)

            if getattr(x, '__ms_mutable__', False):
                return _transform_to(x.__ms_origin_object__)
            elif isinstance(x, ms.Tensor):
                return _transform_to(x)
            elif x in ms.dtype.number_type:
                return _transform_to(x)
            elif isinstance(x, list):
                return list(map(_transform, x))
            elif isinstance(x, tuple):
                return tuple(map(_transform, x))
            elif isinstance(x, dict):
                return {k: _transform(v) for k, v in x.items()}
            else:
                return x

        transformed_op_input, transformed_op_args, transformed_op_kwargs = (
            _transform(self.op_input),
            _transform(self.op_args),
            _transform(self.op_kwargs),
        )

        return OpSampleInput(
            transformed_op_input,
            op_args=transformed_op_args,
            op_kwargs=transformed_op_kwargs,
            op_name=self.op_name + "_transformed_" + method,
        )

    def convert_to_args(self, append_dout=None):
        def _to_args_list(x):
            if isinstance(x, dict):
                return list(x.values())
            elif isinstance(x, (list, tuple)):
                return list(x)
            else:
                return [x]

        op_args = []
        op_args.extend(_to_args_list(self.op_input))
        op_args.extend(_to_args_list(self.op_args))
        op_args.extend(_to_args_list(self.op_kwargs))
        if append_dout is not None:
            op_args.extend(_to_args_list(append_dout))

        return OpSampleInput(
            op_input=None,
            op_args=tuple(op_args),
            op_kwargs={},
            op_name=self.op_name + "_with_dout",
        )

    def copy(self):
        def _copy(x):
            if isinstance(x, ms.Tensor):
                return x.copy()
            elif x in ms.dtype.number_type:
                return x
            elif isinstance(x, list):
                return list(map(_copy, x))
            elif isinstance(x, tuple):
                return tuple(map(_copy, x))
            elif isinstance(x, dict):
                return {k: _copy(v) for k, v in x.items()}
            else:
                return x
        return self.transform(_copy, 'copy')

    def asnumpy(self):
        def _asnumpy(x):
            if isinstance(x, ms.Tensor):
                return ms_asnumpy(x).copy()
            elif x in ms.dtype.number_type:
                return _dtype_to_nptype(x)

            return x

        return self.transform(_asnumpy, 'asnumpy')

    def astorch(
            self,
            *,
            convert_half_to_float: Optional[bool] = False,
            convert_extra_uint: Optional[bool] = False,
    ):
        def _dtype_to_torch_dtype(msdtype):
            msdtype_to_torch_dtype_dict = {
                ms.bool_: torch.bool,
                ms.int8: torch.int8,
                ms.int16: torch.int16,
                ms.int32: torch.int32,
                ms.int64: torch.int64,
                ms.uint8: torch.uint8,
                ms.float16: torch.float16,
                ms.float32: torch.float32,
                ms.float64: torch.float64,
                ms.complex64: torch.complex64,
                ms.complex128: torch.complex128,
                ms.bfloat16: torch.bfloat16,
            }
            return msdtype_to_torch_dtype_dict[msdtype]

        def _astorch(x):
            if isinstance(x, ms.Tensor):
                np_arr = ms_asnumpy(x,
                                    convert_half_to_float=convert_half_to_float,
                                    convert_extra_uint=convert_extra_uint)
                return torch.tensor(np_arr, dtype=torch.bfloat16) if x.dtype == ms.bfloat16 else torch.tensor(np_arr)
            elif x in ms.dtype.number_type:
                return _dtype_to_torch_dtype(x)

            return x

        return self.transform(_astorch, 'astorch')

    def discontiguous(self):
        def _discontiguous(x):
            if isinstance(x, ms.Tensor):
                return _tensor_to_discontiguous(x)
            elif isinstance(x, ms.dtype):
                return x

            return x

        if 'transformed_astorch' in self.op_name:
            raise RuntimeError("OpSampleInput only supports discontiguous method with mindspore.Tensor now.")

        return self.transform(_discontiguous, 'discontiguous')

    def summary(self, values=False):
        def _tensor_summary(x):
            if isinstance(x, (ms.Tensor, torch.Tensor, np.ndarray)):
                sum_info = f"{type(x).__name__}(shape={x.shape}, dtype={x.dtype}"
                if values:
                    sum_info += f", mean={x.mean()}, max={x.max()}, min={x.min()}"
                return sum_info + ")"
            elif isinstance(x, list):
                return f"list[" + ", ".join(map(_tensor_summary, x)) + "]"
            elif isinstance(x, tuple):
                return f"tuple(" + ", ".join(map(_tensor_summary, x)) + ")"
            elif isinstance(x, dict):
                return f"dict(" + ", ".join(f"{k}: {_tensor_summary(v)}" for k, v in x.items()) + ")"
            else:
                return f"{type(x).__name__}({x})"

        return self.__repr__(_tensor_summary)

    def __repr__(self, print_func=lambda x: x):
        return f"OpSampleInput(\n" + \
               f"op_input={print_func(self.op_input)},\n" + \
               f"op_args={print_func(self.op_args)},\n" + \
               f"op_kwargs={print_func(self.op_kwargs)},\n" + \
               f"op_name={print_func(self.op_name)})"


def _tensor_to_discontiguous(x):
    if not x.is_contiguous():
        return x

    if x.numel() == 0 or x.numel() == 1:
        return x

    empty_tensor = x.new_empty(x.shape + (2,))
    if x.dtype == ms.bool:
        empty_tensor[..., 0] = True
    else:
        empty_tensor[..., 0] = math.nan
    empty_tensor[..., 1] = x.copy()
    result = empty_tensor[..., 1]

    assert not result.is_contiguous()
    return result


def ms_asnumpy(tensor, convert_half_to_float=False, convert_extra_uint=False):
    def _sync_host(tensor):
        try:
            host_tensor = tensor.to('cpu')
            _pynative_executor.sync()
            return host_tensor
        except Exception:  # pylint: disable=W0703
            return tensor

    if not isinstance(tensor, ms.Tensor):
        raise ValueError(f"tensor must be a ms.Tensor, but got {type(tensor)}")

    if tensor.dtype == ms.bfloat16:
        return _sync_host(tensor).float().asnumpy()
    if convert_half_to_float and tensor.dtype == ms.float16:
        return _sync_host(tensor).float().asnumpy()
    if convert_extra_uint and tensor.dtype in (ms.uint16, ms.uint32, ms.uint64):
        return _sync_host(tensor).asnumpy().astype(np.int64)
    return _sync_host(tensor).asnumpy()


def make_tensor(
        shape: tuple[int, ...],
        dtype: ms.dtype,
        low: Optional[float] = None,
        high: Optional[float] = None,
        *,
        device: Optional[str] = None,
        discontiguous: Optional[bool] = False,
        random_seed: Optional[int] = None,
        random_method: Optional[str] = None,
):
    def _generate_ndarray(shape, dtype, low, high, random_method):
        def _generate_ndarray_by_random_method(random_method, shape, dtype, low, high):
            if random_method == 'randn':
                ndarray = np.random.randn(*shape)
            elif random_method == 'randint':
                ndarray = np.random.randint(low, high, size=shape)
            elif random_method == 'uniform':
                ndarray = np.random.uniform(low, high, size=shape)
            else:
                raise ValueError(f"Invalid random method: {random_method}")
            if isinstance(ndarray, np.ndarray):
                ndarray = ndarray.astype(dtype)
            return ndarray

        dtype_to_np_dtype_dict = {
            ms.bool: (np.bool_, 'randint', 0, 2),
            ms.int8: (np.int8, 'randint', -9, 10),
            ms.int16: (np.int16, 'randint', -9, 10),
            ms.int32: (np.int32, 'randint', -9, 10),
            ms.int64: (np.int64, 'randint', -9, 10),
            ms.uint8: (np.uint8, 'randint', 0, 10),
            ms.uint16: (np.uint16, 'randint', 0, 10),
            ms.uint32: (np.uint32, 'randint', 0, 10),
            ms.uint64: (np.uint64, 'randint', 0, 10),
            ms.float16: (np.float16, 'randn', None, None),
            ms.float32: (np.float32, 'randn', None, None),
            ms.float64: (np.float64, 'randn', None, None),
            ms.complex64: (np.complex64, 'randn', None, None),
            ms.complex128: (np.complex128, 'randn', None, None),
            ms.bfloat16: (np.float32, 'randn', None, None),
        }
        np_dtype, default_random_method, default_low, default_high = dtype_to_np_dtype_dict[dtype]
        if random_method is None:
            return _generate_ndarray_by_random_method(default_random_method, shape, np_dtype, default_low, default_high)
        else:
            if random_method != 'randn':
                assert low is not None and high is not None, "low and high must be specified for non-randn method."
            return _generate_ndarray_by_random_method(random_method, shape, np_dtype, low, high)

    if random_seed is not None:
        np.random.seed(random_seed)

    if dtype == ms.complex64 or dtype == ms.complex128:
        real = _generate_ndarray(shape, dtype, low, high, random_method)
        imag = _generate_ndarray(shape, dtype, low, high, random_method)
        result = ms.Tensor((real + 1j * imag), dtype=dtype)
    else:
        result = ms.Tensor(_generate_ndarray(shape, dtype, low, high, random_method), dtype=dtype)

    if device is not None:
        result = result.to(device)

    if discontiguous:
        result = _tensor_to_discontiguous(result)

    return result


def make_tensor_with_np_array(
        np_array: np.ndarray,
        dtype=None,
        *,
        device: Optional[str] = None,
        discontiguous: Optional[bool] = False,
):
    result = ms.Tensor(np_array, dtype=dtype)

    if device is not None:
        result = result.to(device)
    if discontiguous:
        result = _tensor_to_discontiguous(result)

    return result
