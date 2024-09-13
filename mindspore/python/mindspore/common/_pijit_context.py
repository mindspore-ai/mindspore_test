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
"""Define pijit context"""

import inspect
import types
import functools
import importlib.util
import mindspore
from mindspore import log as logger
from mindspore.common.jit_config import JitConfig
from mindspore._c_expression import GraphExecutor_, jit_mode_pi_enable, jit_mode_pi_disable, pi_jit_set_context


def _update_graph_executor_config(jit_config):
    """Update GraphExecutor jit_config"""
    if isinstance(jit_config, JitConfig):
        jit_config = jit_config.jit_config_dict
    if not isinstance(jit_config, dict):
        return
    valid_config = {}
    for k, v in jit_config.items():
        valid_config[str(k)] = str(v)
    GraphExecutor_.get_instance().set_jit_config(JitConfig(**valid_config).jit_config_dict)


class PIJitCaptureContext:
    """
    Context manager for pijit graph capture
    """

    def __init__(self, jit_config=None, input_signature=None):
        _update_graph_executor_config(jit_config)
        config = {}
        if isinstance(jit_config, JitConfig):
            config.update(jit_config.jit_config_dict)
        elif jit_config is not None:
            config.update(jit_config)

        self.config = config
        self.input_signature = input_signature
        self.ret = None
        self.fn = None
        self._init_arg = iter([self.config, self.input_signature])

        if not SKIP_RULES:
            return
        pi_jit_set_context(wrapper=self._wrapper(),
                           skip_files=_get_skip_files(),
                           skip_codes=SKIP_RULES["codes"])
        SKIP_RULES.clear()

    @staticmethod
    def _is_unsupported(fn):
        # generator, coroutine, awaitable and a function that return them is unsupported
        return inspect.isgeneratorfunction(fn) or inspect.iscoroutinefunction(fn) \
            or inspect.isasyncgenfunction(fn) or inspect.isawaitable(fn)

    def _wrapper(self):
        def _fn(*args, **kwds):
            with self:
                self.ret = self.fn(*args, **kwds)
                return self.ret
        return _fn

    def __call__(self, fn):
        if isinstance(fn, type) and issubclass(fn, mindspore.nn.Cell):
            fn.construct = self(fn.construct)
            return fn
        if isinstance(fn, mindspore.nn.Cell):
            type(fn).construct = self(type(fn).construct)
            return fn
        if isinstance(fn, types.MethodType):
            return types.MethodType(self(fn.__func__), fn.__self__)
        if not isinstance(fn, types.FunctionType) or self._is_unsupported(fn):
            logger.warning("unsupported function type" + str(fn))
            return fn

        try:
            if inspect.getmodule(fn.__code__).__name__.startswith("mindspore"):
                return fn
        finally:
            pass

        _fn = self._wrapper()
        if fn.__code__ is _fn.__code__:
            fn = fn.__closure__[0].cell_contents.fn
        self.fn = fn
        return functools.wraps(fn)(_fn)

    def __enter__(self):
        pi_jit_set_context(self.fn, *self._init_arg)
        jit_mode_pi_enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pi_jit_set_context(None)
        jit_mode_pi_disable()


def _get_skip_files():
    """
    Get skip files by SKIP_RULES
    """
    def _filter(path: str):
        if path.endswith("__init__.py"):
            return path[0:-11]
        return path

    # not import these modules, only find it
    find = importlib.util.find_spec

    files = [*SKIP_RULES["skip_dirs"]]
    files += [_filter(find(m).origin) for m in SKIP_RULES["builtins"]]
    for i in SKIP_RULES["third_party"]:
        spec = find(i)
        if spec is None:
            continue
        files.append(_filter(spec.origin))

    return tuple(files)


# complete the skip list...
SKIP_RULES = {
    "skip_dirs": (
        "<frozen importlib",
        "<__array_function__ internals>",
        "<string>",
    ),
    "builtins": (
        "mindspore",  # not capture any function of mindspore unless it's called by user
        "abc",
        "ast",
        "codecs",
        "collections",
        "contextlib",
        "copy",
        "copyreg",
        "dataclasses",
        "enum",
        "functools",
        "glob",
        "importlib",
        "inspect",
        "linecache",
        "logging",
        "multiprocessing",
        "operator",
        "os",
        "posixpath",
        "random",
        "re",
        "selectors",
        "signal",
        "tempfile",
        "threading",
        "tokenize",
        "traceback",
        "types",
        "typing",
        "unittest",
        "weakref",
        "_collections_abc",
        "_weakrefset",
        # others...
        "sre_compile",
        "sre_parse",
        "genericpath",
    ),
    "third_party": (
        "numpy",
        "pandas",
        "sklearn",
        "tqdm",
        "tree",
    ),
    "codes": (),
}
