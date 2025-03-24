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

"""JIT Context for trace JIT."""

import inspect
import re
from functools import wraps
import mindspore as ms
from mindspore import log as logger
from mindspore import context
from mindspore.common.jit_context import JitContext, set_jit_context, jit_context
from mindspore.common.tensor import Tensor as PythonTensor
from mindspore._checkparam import is_stub_tensor
from mindspore._c_expression import TraceRecorder as tr
from mindspore._c_expression import JitExecutor_
from mindspore._c_expression import TensorNode
from mindspore._c_expression import TensorPy as Tensor, CSRTensor, COOTensor
from mindspore._c_expression import typing


class TraceJitContext(JitContext):
    """JIT Context for trace JIT."""

    def __init__(self):
        JitContext.__init__(self)
        self._is_nested = False

    def set_is_nested(self, status):
        self._is_nested = status

    def is_nested(self):
        return self._is_nested

    def run_op(self, prim, prim_res, *args):
        """Capture op"""
        logger.debug(f'prim: {prim}, args: {args}, prim_res: {prim_res}')
        if isinstance(prim_res, TensorNode):
            prim_res = prim_res.get_value()
        prim_res = _sync_stub_tensor(prim_res)
        args = tuple(_sync_stub_tensor(arg) for arg in args)
        args = tuple(_convert_arg_for_operators(arg, prim.name) for arg in args)
        file_names, linenos = _get_caller_lines()
        tr.get_instance().new_node(prim, prim_res, file_names, linenos, False, *args)
        return prim_res

    def run_graph(self, phase, prim_res, *args):
        """Capture func_graph generated from ast"""
        logger.debug(f'phase: {phase}, args: {args}, prim_res: {prim_res}')
        if isinstance(prim_res, TensorNode):
            prim_res = prim_res.get_value()
        prim_res = _sync_stub_tensor(prim_res)
        args = tuple(_sync_stub_tensor(arg) for arg in args)
        file_names, linenos = _get_caller_lines()
        tr.get_instance().new_fg_node((phase, prim_res, file_names, linenos, self._is_nested), *args)
        return prim_res


_compile_only = False
_trace_jit_context = TraceJitContext()
_trace_compile_cache = set()
_jit_executor = JitExecutor_.get_instance()
_using_trace = False


def _set_compile_only(compile_only=True):
    global _compile_only
    _compile_only = compile_only


def _sync_stub_tensor(stub):
    """Synchronize stub tensor"""
    if is_stub_tensor(stub):
        real_tensor = stub.stub_sync()
        logger.debug(f'Convert stub tensor, stub: [{type(stub)}] {id(stub)}/{stub}, '
                     f'tensor: [{type(real_tensor)}] {id(real_tensor)}/{real_tensor}')
        return real_tensor
    if isinstance(stub, tuple):
        return tuple(_sync_stub_tensor(item) for item in stub)
    if isinstance(stub, list):
        return list(_sync_stub_tensor(item) for item in stub)
    return stub


def convert_tensorpy(args):
    new_args = []
    for arg in args:
        if isinstance(arg, Tensor):
            new_args.append(PythonTensor(arg))
        else:
            new_args.append(arg)
    return tuple(new_args)


def _convert_arg_for_operators(arg, prim_name):
    """Convert dtype to enum"""
    from mindspore.ops._utils.arg_dtype_cast import DtypeToEnum
    if isinstance(arg, typing.Type):
        return DtypeToEnum()(prim_name, 'dtype', arg)
    return arg



def nested_run(obj, cell, *args):
    """Start a trace process nested in ast."""
    set_jit_context(_trace_jit_context)
    _trace_jit_context.set_is_nested(True)
    args = args[0]
    args = convert_tensorpy(args)
    if isinstance(obj, ms.nn.Cell):
        res = obj.construct.__wrapped__(obj, *args)
    else:
        if not cell:
            res = obj.__wrapped__(*args)
        else:
            res = obj.__wrapped__(cell, *args)
    if res is not tuple:
        res = (res,)
    file_names, linenos = _get_caller_lines()
    res = _sync_stub_tensor(res)
    set_jit_context(None)
    return file_names, linenos, res


def _jit_trace(fn):
    """
    Create a callable MindSpore graph from a Python function by trace method.

    This allows the MindSpore runtime to apply optimizations based on traced func graph.

    Args:
        fn (Function): The Python function that will be run as a graph. Default: ``None`` .

    Returns:
        Function, if `fn` is not None, returns a callable function that will execute the compiled function; If `fn` is
        None, returns a decorator and when this decorator invokes with a single `fn` argument, the callable function is
        equal to the case when `fn` is not None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.common.jit_trace import _jit_trace as jit_trace
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        ...
        >>> # To create a callable MindSpore graph by calling decorator @jit_trace
        >>> def tensor_add(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> tensor_add_graph = jit_trace(fn=tensor_add)
        >>> out = tensor_add_graph(x, y)
    """

    @wraps(fn)
    def jit_trace_wrap(*args, **kwargs):
        # If a trace graph is already built, keep going without building a new trace graph.
        if jit_context():
            return fn(*args, **kwargs)
        # Start trace process.
        if kwargs:
            bound_arguments = inspect.signature(fn).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args = bound_arguments.args
            kwargs = bound_arguments.kwargs
        generate_name = fn.__module__
        if args:
            jit_args = args[1:] if hasattr(args[0], fn.__name__) else args
            obj = args[0]
            if hasattr(obj, fn.__name__):  # Add class name for Cell.
                generate_name = generate_name + "." + obj.__class__.__name__
        else:
            jit_args = args
        generate_name = generate_name + "." + fn.__name__ + "#" + str(id(fn))
        # Add create time for Cell.
        if args and hasattr(obj, fn.__name__):
            generate_name = generate_name + '#created_' + str(args[0].create_time)
        line_str = fn.__code__.co_filename + ":" + str(fn.__code__.co_firstlineno)
        generate_name = generate_name + '#[' + line_str + ']'

        new_compile = _jit_trace_begin(generate_name, *jit_args)
        if new_compile:
            fn_res = fn(*args, **kwargs)
            logger.debug(f'fn: {fn}, fn_res: {fn_res}, line: {line_str}')
            # Use fn's output to build func graph's output.
            output = _jit_trace_end(fn_res)
        else:
            output = _jit_trace_end(None)  # Run with compilation.
        logger.debug(f'output: {output}')
        return output

    jit_trace_wrap.__trace_func__ = True
    return jit_trace_wrap


def _get_caller_lines():
    """Get caller code line info."""
    file_names = []
    linenos = []
    for frame_info in inspect.stack():
        logger.debug(f'\t- frame: {frame_info[1]}:{frame_info[2]}/{frame_info[4][0]}')
        file_name = frame_info[1]
        if re.search(r'mindspore/common/.*\.py|mindspore/ops/.*\.py|mindspore/nn/.*\.py', file_name) is not None:
            continue
        lineno = frame_info[2]
        logger.debug(f'Match caller frame: {frame_info[1]}:{frame_info[2]}/{frame_info[4][0]}')
        file_names.append(file_name)
        linenos.append(lineno)
    return file_names, linenos


def _get_args_for_run(args):
    """Get the actual input args and kwargs for runtime."""
    new_args = []
    for arg in args:
        if isinstance(arg, PythonTensor):
            if arg.has_init:
                arg.init_data()
            if not arg.const_arg:
                new_args.append(arg)
        elif isinstance(arg, (Tensor, CSRTensor, COOTensor)):
            new_args.append(arg)
        elif context.get_context("grad_for_scalar") and isinstance(arg, (int, float)):
            new_args.append(arg)
        elif isinstance(arg, dict) and hasattr(arg, "__ms_mutable__"):
            new_args.append(tuple(arg.values()))
    return tuple(new_args)


def _jit_trace_begin(fn_name, *args):
    """
    Start to build a MindIR func graph for a code snippet by trace method.

    This allows the MindSpore runtime to apply optimizations based on traced func graph.

    Note:
        Use it with `_jit_trace_end` cooperatively.

    Also see: :func:`_jit_trace_end`.

    Args:
        fn_name (str): The name of func graph to be built.
        args (tuple): The arguments of func graph.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.common.jit_trace import _jit_trace_begin, _jit_trace_end
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> def tensor_add(x, y):
        ...     _jit_trace_begin(x, y)
        ...     z = x + y
        ...     z = _jit_trace_end(z)
        ...     return z
        ...
        >>> out = tensor_add(x, y)
    """
    global _using_trace
    if _using_trace:
        raise RuntimeError(
            "Should not use jit_block and jit_trace at the same time.")
    _using_trace = True
    logger.debug(f'_jit_trace_begin, args: {args}')
    _trace_jit_context.set_is_nested(False)
    set_jit_context(_trace_jit_context)
    args = tuple(_sync_stub_tensor(arg) for arg in args)
    for arg in args:
        logger.debug(f'_jit_trace_begin, arg: {arg}, {type(arg)}')

    # Generate phase for compile pipeline.
    key = _jit_executor.generate_arguments_key(None, args, dict(), False)
    from mindspore.common.api import _PyNativeExecutor
    phase = fn_name + '.' + str(key)
    if _PyNativeExecutor().requires_grad():
        phase = phase + ".grad"
    logger.debug(f'phase: {phase}')
    # Compiled before, just run.
    if not _compile_only and phase in _trace_compile_cache:
        logger.debug('Had compiled, just run.')
        _trace_jit_context.compiled = True
        output = tr.get_instance().run_graph(phase, args)
        from mindspore.common.api import _convert_python_data
        _trace_jit_context.result = _convert_python_data(output)
        logger.debug(f'jit trace result: {_trace_jit_context.result}')
        return False
    logger.debug('Start compiling...')
    file_names, linenos = _get_caller_lines()
    fn_short_name = fn_name.split('#')[0]
    tr.get_instance().begin_graph(fn_short_name, phase, file_names, linenos, *args)
    _trace_compile_cache.add(phase)
    # Save for first call, used in end().
    _trace_jit_context.phase = phase
    _trace_jit_context.args = args
    return True


def _jit_trace_end(*output_args):
    """
    Finish building a MindIR func graph for a code snippet by trace method.

    This allows the MindSpore runtime to apply optimizations based on traced func graph.

    Note:
        Use it with `_jit_trace_begin` cooperatively.

    Also see: :func:`_jit_trace_begin`.

    Args:
        output_args (tuple): The output of func graph.

    Returns:
        The same as args `output_args`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.common.jit_trace import _jit_trace_begin, _jit_trace_end
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> def tensor_add(x, y):
        ...     _jit_trace_begin(x, y)
        ...     z = x + y
        ...     z = _jit_trace_end(z)
        ...     return z
        ...
        >>> out = tensor_add(x, y)
    """
    if _trace_jit_context.compiled:
        output = _trace_jit_context.result
        logger.debug(f'jit trace result: {output}')
    else:
        logger.debug(f'output_args: {output_args}')
        output_args = tuple(_sync_stub_tensor(arg) for arg in output_args)
        file_names, linenos = _get_caller_lines()
        tr.get_instance().end_graph(file_names, linenos, *output_args)
        if _compile_only:
            output = output_args[0] if len(output_args) == 1 else output_args
        else:
            args = _get_args_for_run(_trace_jit_context.args)
            output = tr.get_instance().run_graph(
                _trace_jit_context.phase, args)
            from mindspore.common.api import _convert_python_data
            output = _convert_python_data(output)
            logger.debug(f'jit trace result: {output}')
            logger.debug(
                f'python result: {output_args[0] if len(output_args) == 1 else output_args}')
            _trace_jit_context.phase = ''
            _trace_jit_context.args = None
    set_jit_context(None)
    _trace_jit_context.compiled = False
    global _using_trace
    _using_trace = False
    return output
