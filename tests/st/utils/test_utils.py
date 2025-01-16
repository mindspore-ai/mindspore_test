# Copyright 2023 Huawei Technologies Co., Ltd
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

from functools import wraps
import inspect
import sys
from typing import Sequence

import numpy as np

import mindspore as ms
from mindspore import jit, nn, Tensor

if sys.version_info >= (3, 9):
    list_annotation = list
else:
    from typing import List
    list_annotation = List

ms.set_context(jit_syntax_level=ms.STRICT)


class Net(nn.Cell):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def construct(self, *inputs, **kwargs):
        return self.func(*inputs, **kwargs)


def run_with_cell(fn):
    if fn is None:
        raise ValueError("fn cannot be none!")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        cell_obj = Net(fn)
        return cell_obj(*args, **kwargs)

    return wrapper


def run_with_mode(fn):
    if fn is None:
        raise ValueError("fn cannot be none!")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'mode' not in kwargs:
            raise ValueError("mode not provided.")
        mode = kwargs['mode'].lower()
        if mode not in ['pynative', 'graph', 'kbk']:
            raise ValueError(
                "Invalid mode. Available option: ['pynative', 'graph', 'kbk'].")

        del kwargs['mode']
        if mode == "graph":
            return (jit(fn, backend="GE"))(*args, **kwargs)
        if mode == "kbk":
            return (jit(fn, jit_level="O0"))(*args, **kwargs)
        return fn(*args, **kwargs)

    setattr(wrapper, "__wrapped_with_mode__", True)
    return wrapper


def run_with_cell_ext(jit_config=None):
    def cell_wrap_fn(fn):
        if fn is None:
            raise ValueError("fn cannot be none!")

        @wraps(fn)
        def wrapper(*args, **kwargs):
            cell_obj = Net(fn)
            if jit_config:
                cell_obj.set_jit_config(jit_config)
            return cell_obj(*args, **kwargs)

        return wrapper

    return cell_wrap_fn


def to_cell_obj(fn):
    cell_obj = Net(fn)
    return cell_obj


def compare(output, expect):
    '''
    :param output: Tensor, including tuple/list of tensor
    :param expect: Numpy array, including tuple/list of Numpy array
    :return:
    '''
    if isinstance(output, (tuple, list)):
        for o_ele, e_ele in zip(output, expect):
            compare(o_ele, e_ele)
    else:
        if expect.dtype == np.float32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 1e-3, 1e-3
        if not np.allclose(output.asnumpy(), expect, rtol, atol, equal_nan=True):
            raise ValueError(f"compare failed \n output: {output.asnumpy()}\n expect: {expect}")


def generate_random_input(shape: Sequence[int], dtype: type = None) -> np.ndarray:
    array = np.random.randn(*shape)
    if dtype:
        array = array.astype(dtype)
    return array


def generate_random_tensor(shape: Sequence[int], dtype: ms.dtype) -> ms.Tensor:
    # Q: Why use `numpy.random.randn` to generate a random `numpy.ndarray` and then convert it into a
    #    `mindspore.Tensor` instead of directly using `mindspore.ops.StandardNormal` to generate a random
    #    `mindspore.Tensor`?
    # A: Because `mindspore.ops.StandardNormal` does not support the random seed reproduction function on the Ascend
    #    backend, which is not conducive to reproduct results. Reference
    #    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.StandardNormal.html .
    return ms.Tensor(generate_random_input(shape)).type(dtype)


def get_inputs_np(shapes, dtypes):
    np.random.seed(10)
    inputs_np = []
    for shape, dtype in zip(shapes, dtypes):
        inputs_np.append(generate_random_input(shape, dtype))
    return inputs_np


def get_inputs_tensor(inputs_np):
    inputs = []
    for input_np in inputs_np:
        inputs.append(Tensor(input_np))
    return inputs


def convert_ms_tensor_to_numpy_array(tensor: ms.Tensor) -> np.ndarray:
    if tensor.dtype == ms.bfloat16:
        tensor = tensor.astype(ms.float32)
    return tensor.asnumpy()


def convert_ms_tensors_to_numpy_arrays(tensors: Sequence[ms.Tensor]) -> list_annotation[np.ndarray]:
    return [convert_ms_tensor_to_numpy_array(tensor) for tensor in tensors]


def need_run_graph_op_mode(func, args, kwargs):
    if ms.get_context('device_target') != 'Ascend':
        return False

    # get description of function params expected
    sig = inspect.signature(func)
    sig_args = [param.name for param in sig.parameters.values()]

    mode = None
    if isinstance(kwargs, dict):
        for key in ['mode', 'context_mode']:
            if key in sig_args and key in kwargs:
                mode = kwargs[key]
                break

    return mode == ms.GRAPH_MODE


def run_test_with_On(test_func):

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # call original test function
        test_func(*args, **kwargs)

        if not need_run_graph_op_mode(test_func, args, kwargs):
            return

        org_jit_level = ms.get_context('jit_level')
        try:
            # run graph in kernel by kernel mode
            ms.set_context(jit_level='O0')
            test_func(*args, **kwargs)
        finally:
            ms.set_context(jit_level=org_jit_level)

    return wrapper
