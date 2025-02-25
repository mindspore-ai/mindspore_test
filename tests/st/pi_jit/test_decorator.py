# coding=utf-8

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
""" test decorator """
import functools
from mindspore import context, jit, Tensor, ops

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode
from tests.st.pi_jit.share.utils import pi_jit_with_config

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_decorator_and_context_manager():
    """
    Feature: decorator.
    Description: decorator + context manager, and a subgraph returns free variable.
    Expectation: no graph break.
    """

    class ContextDecorator(object):
        def _recreate_cm(self):
            return self  # `self` is a free variable, and it is the return value of this subgraph

        def __call__(self, func):
            @functools.wraps(func)
            def inner(*args, **kwds):
                with self._recreate_cm():  # `self` is a free variable.
                    return func(*args, **kwds)

            return inner

    class no_grad(ContextDecorator):
        def __init__(self):
            super().__init__()
            self.prev_state = False

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    @no_grad()
    def func_2(x: Tensor):
        return ops.add(x, x)

    def fn(x: Tensor):
        return func_2(x)

    a = Tensor([1, 2, 3])
    o1 = fn(a)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(a)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)
