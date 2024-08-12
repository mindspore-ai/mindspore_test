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
"""test call function ex implement"""
import sys
import pytest
import mindspore.context as context
from tests.mark_utils import arg_mark
from mindspore import Tensor, jit, JitConfig
from mindspore._c_expression import update_pijit_default_config, get_code_extra


@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping tests on Python 3.11 and higher.")


SYS_VER = (sys.version_info.major, sys.version_info.minor)
if SYS_VER != (3, 7) and SYS_VER != (3, 9):
    pytest.skip(reason="not implement for python" + str(SYS_VER), allow_module_level=True)


def add(a, b):
    return a + b


@jit(mode="PIJit")
def add_tuple(a, b):
    c = (a, b)
    return add(*c)


@jit(mode="PIJit")
def add_list(a, b):
    c = [a, b]
    return add(*c)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [add_list, add_tuple])
def test_call_ex_param(jit_func):
    """
    Feature: call ex param implement
    Description: test CALL_FUNCTION_EX.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    assert(all(jit_func(x, y) == Tensor([3])))
    assert(get_code_extra(jit_func.__wrapped__)['break_count_'] == 0)
