# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""Define dynamic_tensor_shapes decorator."""
import types
import inspect
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.common._utils import setattr_with_func, getattr_with_func


DYNAMIC_TENSOR_SHAPES = "__dynamic_tensor_shapes__"


class TensorShape:
    """Define TensorShape used for dynamic tensor shapes."""
    def __init__(self, shape):
        self.shape = get_shape(shape)

    @staticmethod
    """Check elements in shape."""
    def check_element_valid(item):
        return item is None or (isinstance(item, int) and item > 0)

    @staticmethod
    """Check and get shape."""
    def get_shape(shape):
        if shape is None:
            return shape
        if isinstance(shape, (tuple, list)) and None in shape and all(check_element_valid(item) for item in shape):
            return shape
        raise TypeError(f"Invalid shape '{shape}'. TensorShape only supports None " \
                        f"or a tuple/list of positive integers and None.")

def _check_shape_valid(shape):
    """Check if arg is valid."""
    if isinstance(shape, TensorShape):
        return True
    if isinstance(shape, (tuple, list)) and all(_check_shape_valid(item) for item in shape):
        return True
    raise TypeError(f"The decorator dynamic_tensor_shapes only supports TensorShape or a tuple/list of TensorShape.")


def _check_arg_shape_and_dtype(arg, name, dyn_shape):
    """Check the shape and dtype of argument."""
    if not isinstance(arg, Tensor):
        raise TypeError(f"When using decorator dynamic_tensor_shapes, the corresponding inputs should be " \
                        f"of type Tensor, but '{name}' is of type {type(arg)}.")
    if dyn_shape is not None:
        arg_shape = arg.shape
        if len(arg_shape) != len(dyn_shape) or any(y is not None and x != y for x, y in zip(arg_shape, dyn_shape)):
            raise ValueError(f"When using decorator dynamic_tensor_shapes, the shape of argument '{name}' " \
                             f"should match {dyn_shape}, but got {arg_shape}.")


def generate_dynamic_tensor_args(args_list, dynamic_shapes):
    """Generate compile args with dynamic_shapes"""
    new_compile_args = list(args_list)
    for index, arg in enumerate(args_list):
        if index not in dynamic_shapes:
            continue
        name, dyn_shape = dynamic_shapes[index]
        _check_arg_shape_and_dtype(arg, name, dyn_shape)
        new_compile_args[index] = Tensor(shape=dyn_shape, dtype=arg.dtype)
    logger.debug(f"args_list: {args_list}, dynamic_shapes: {dynamic_shapes}, " \
                 f"new_compile_args: {new_compile_args}")
    return new_compile_args


def dynamic_tensor_shapes(*args, **kwargs):
    """Define dynamic_tensor_shapes decorator"""
    # Check inputs at first.
    if args:
        raise ValueError(f"Decorator dynamic_tensor_shapes only supports keyword arguments as inputs, but got {args}.")
    for arg in kwargs.values():
        _check_shape_valid(arg)

    def decorator(func):
        if not isinstance(func, (types.FunctionType, types.MethodType)):
            raise ValueError(f"Decorator dynamic_tensor_shapes can only be used for function or method " \
                             f"decrocated by ms.jit, but got {func}.")
        signature = inspect.signature(func)
        sigs_name = [sig_name for sig_name in signature.parameters if sig_name != "self"]
        if len(kwargs) > len(sigs_name):
            raise ValueError(f"When using decorator dynamic_tensor_shapes, the number of arguments {len(kwargs)} " \
                             f"exceeds the number of function arguments {len(sigs_name)}.")
        # Generate dynamic args.
        dynamic_args = dict()
        for key, value in kwargs.items():
            index = sigs_name.index(key)
            if index in dynamic_args:
                raise ValueError(f"keyword argument repeated: {key}")
            dynamic_args[index] = (key, value)
        # Set dynamic_tensor_shape to func.
        inner_func = inspect.unwrap(func, stop=lambda f: not hasattr(f, '__wrapped__'))
        setattr_with_func(inner_func, DYNAMIC_TENSOR_SHAPES, dynamic_args)
        logger.info(f"Set dynamic tensor shapes: {dynamic_args} to {inner_func}")
        return func
    return decorator
