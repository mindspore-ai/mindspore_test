# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Store the deprecated api temporarily."""
from __future__ import absolute_import

import os
import types
import time
from functools import wraps
from mindspore import log as logger
from mindspore.common.tensor import Tensor as PythonTensor
from mindspore.common.api import _get_jit_hash, _process_dyn_args, _handle_func_args, _JitExecutor
from mindspore.parallel._utils import _is_pynative_parallel
from mindspore._c_expression.amp import get_curr_amp_strategy
from mindspore.common._pijit_context import PIJitCaptureContext


_PYNATIVE_PARALLEL_FUNC_NAME = "after_shard"


def jit(fn=None, mode="PSJit", input_signature=None, hash_args=None, jit_config=None, compile_once=False):
    """
    Create a callable MindSpore graph from a Python function.

    This allows the MindSpore runtime to apply optimizations based on graph.

    Note:
        - If `input_signature` is specified, each input of `fn` must be a Tensor. And the input arguments for `fn`
          will not accept `**kwargs`.
        - It is not supported to run a function with decoration @jit(mode=“PIJit”)
          in static graph mode, in which case the decoration @jit(mode=“PIJit”) is considered invalid.
        - Calls to functions with decorated @jit(mode=“PIJit”) inside functions
          decorated with @jit(mode=“PIJit”) are not supported,
          and the decoration @jit(mode=“PIJit”) is considered invalid.

    Args:
        fn (Function): The Python function that will be run as a graph. Default: ``None`` .
        mode (str): The type of jit used, the value of mode should be ``PIJit`` or ``PSJit``. Default: ``PSJit`` .

            - PSJit:
              Parse python ast to build graph.
            - PIJit:
              Parse python bytecode to build graph at runtime.

        input_signature (Union[Tuple, List, Dict, Tensor]): The Tensor which describes the input arguments. The
            shape and dtype of the Tensor will be supplied to this function. If `input_signature` is specified, the
            input parameters of `fn` cannot accept `**kwargs`, and the shape and dtype of actual inputs should keep the
            same as `input_signature`. Otherwise, TypeError will be raised. There are two mode for `input_signature`:

            - Full mode: Arguments is a Tuple, List or a Tensor, and they will be used as all compile inputs
              for graph-compiling.
            - Incremental mode: Argument is a Dict, and they will set to some of the graph inputs, which will be
              substituted into the input at the corresponding position for graph-compiling.

            Default: ``None`` .

        hash_args (Union[Object, List or Tuple of Objects]): The local free variables used inside `fn`,
            like functions or objects of class defined outside `fn`. Calling `fn` again with change of `hash_args`
            will trigger recompilation. Default: ``None`` .
        jit_config (JitConfig): Jit config for compile. Default: ``None`` .
        compile_once(bool): ``True``: The function would be compiled once when it was created many times.
            But it may be wrong if the free variables were changed. ``False`` : It would be recompiled when
            it was created again.
            Default: ``False`` .

    Returns:
        Function, if `fn` is not None, returns a callable function that will execute the compiled function; If `fn` is
        None, returns a decorator and when this decorator invokes with a single `fn` argument, the callable function is
        equal to the case when `fn` is not None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import mindspore._deprecated.jit as jit
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        ...
        >>> # create a callable MindSpore graph by calling decorator @jit
        >>> def tensor_add(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> tensor_add_graph = jit(fn=tensor_add)
        >>> out = tensor_add_graph(x, y)
        ...
        >>> # create a callable MindSpore graph through decorator @jit
        >>> @jit
        ... def tensor_add_with_dec(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_with_dec(x, y)
        ...
        >>> # create a callable MindSpore graph through decorator @jit with input_signature parameter
        >>> @jit(input_signature=(Tensor(np.ones([1, 1, 3, 3]).astype(np.float32)),
        ...                       Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))))
        ... def tensor_add_with_sig(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_with_sig(x, y)
        ...
        >>> @jit(input_signature={"y": Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))})
        ... def tensor_add_with_sig_1(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out1 = tensor_add_with_sig_1(x, y)
        ...
        ... # Set hash_args as fn, otherwise cache of compiled closure_fn will not be reused.
        ... # While fn differs during calling again, recompilation will be triggered.
        >>> def func(x):
        ...     return ops.exp(x)
        ...
        >>> def closure_fn(x, fn):
        ...     @jit(hash_args=fn)
        ...     def inner_fn(a):
        ...         return fn(a)
        ...     return inner_fn(x)
        ...
        >>> inputs = Tensor(np.ones([10, 10, 10]).astype(np.float32))
        >>> for i in range(10):
        ...     closure_fn(inputs, func)
        ...
        ... # Set compile_once = True, otherwise the train_step will be compiled again.
        >>> def train(x):
        ...     @jit(compile_once = True)
        ...     def train_step(x):
        ...         return ops.exp(x)
        ...     for i in range(10):
        ...         train_step(x)
        ...
        >>> inputs = Tensor(np.ones([10, 10, 10]).astype(np.float32))
        >>> for i in range(10):
        ...     train(inputs)
    """

    def wrap_mindspore(func):
        if not isinstance(compile_once, bool):
            logger.warning(f"The parameter `compile_once` of jit should be a bool, "
                           f"but got {type(compile_once)}.")
        if hash_args:
            hash_obj = _get_jit_hash(hash_args)
        elif compile_once:
            hash_obj = 0
        else:
            hash_obj = int(time.time() * 1e9)

        dyn_args = _process_dyn_args(func, input_signature)

        @wraps(func)
        def staging_specialize(*args, **kwargs):
            if os.getenv("MS_JIT") == '0':
                return func(*args, **kwargs)

            args, kwargs = _handle_func_args(func, *args, **kwargs)

            process_obj = None
            if args and not isinstance(args[0], PythonTensor) and hasattr(args[0], func.__name__):
                process_obj = args[0]
            # only the function or cell instance wrapped by shard will fall into this branch
            if _is_pynative_parallel() and func.__name__ == _PYNATIVE_PARALLEL_FUNC_NAME:
                process_obj = hash_args
            # Handle auto mixed precision strategy.
            if not hasattr(func, "amp_strategy"):
                if isinstance(func, types.MethodType):
                    setattr(func.__func__, "amp_strategy", get_curr_amp_strategy())
                else:
                    setattr(func, "amp_strategy", get_curr_amp_strategy())
            out = _JitExecutor(func, hash_obj, dyn_args, process_obj, jit_config)(*args, **kwargs)
            return out

        return staging_specialize

    wrap_func = wrap_mindspore
    if mode == "PIJit":
        wrap_func = PIJitCaptureContext(jit_config, input_signature)

    if fn is not None:
        return wrap_func(fn)
    return wrap_func
