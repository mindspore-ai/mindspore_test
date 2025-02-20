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
""" test builtin dict """
import pytest
from mindspore import context, jit, Tensor, ops

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_equal, assert_no_graph_break, assert_executed_by_graph_mode, \
    assert_has_graph_break
from tests.st.pi_jit.share.utils import pi_jit_with_config

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_empty_dict():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create an empty dict.
    Expectation: no graph break.
    """

    def fn(x: Tensor):
        d = dict()
        return d, ops.add(x, 1)

    a = Tensor([1, 2, 3])
    o1 = fn(a)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(a)

    assert_equal(o1, o2)
    assert_no_graph_break(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_list():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create an empty dict.
    Expectation: no graph break.
    """

    def fn(kv: list):
        d = dict(kv)
        return ops.add(d[0], d[1])

    kv = [[0, Tensor([1, 2, 3])], [1, Tensor([2, 3, 4])]]
    o1 = fn(kv)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(kv)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_tuple():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: no graph break.
    """

    def fn(kv: tuple):
        d = dict(kv)
        return ops.sub(d[0], d[1])

    kv = ((0, Tensor([1, 2, 3])), (1, Tensor([2, 3, 4])))
    o1 = fn(kv)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(kv)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="Need Fixed")
def test_create_dict_from_duplicate_keys():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: no graph break.
    """

    def fn(kv: list):
        d = dict(kv)
        return ops.sub(d[0], d[1])

    kv = [(0, Tensor([1, 2])), (1, Tensor([2, 3])), (0, Tensor([3, 4])), (2, Tensor([4, 5])), (1, Tensor([5, 6])),
          (0, Tensor([6, 7]))]
    o1 = fn(kv)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(kv)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_dict():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: no graph break.
    """

    def fn(d: dict):
        d2 = dict(d)
        return ops.mul(d2[0], d2[1])

    d = {0: Tensor([1., 2., 3.]), 1: Tensor([3, 4, 5])}
    o1 = fn(d)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(d)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_zip():
    """
    Feature: builtin dict.
    Description: use dict() constructor to create a dict.
    Expectation: graph break, but result is correct.
    """

    def fn(seq1, seq2):
        d2 = dict(zip(seq1, seq2))  # dict(zip(...)) is unsupported, graph break
        return ops.mul(d2['a'], d2['b'])

    a = ['a', 'b']
    b = [Tensor([1, 2, 3]), Tensor([2, 3, 4])]
    o1 = fn(a, b)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(a, b)

    assert_equal(o1, o2)
    assert_has_graph_break(compiled_fn, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_dict_from_kwargs():
    """
    Feature: builtin dict.
    Description: use dict(**kwargs) constructor to create a dict.
    Expectation: graph break, but result is correct.
    """

    def fn(v: Tensor):
        d2 = dict(a=v)  # dict(**kwargs) is unsupported, graph break
        return ops.mul(d2['a'], 2)

    a = Tensor([1, 2, 3])
    o1 = fn(a)

    compiled_fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = compiled_fn(a)

    assert_equal(o1, o2)
    assert_has_graph_break(compiled_fn, break_count=1)
