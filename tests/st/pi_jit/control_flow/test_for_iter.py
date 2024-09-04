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
''' test FOR_ITER for pijit '''
import sys
import pytest
import dis
import sys
from mindspore import jit, Tensor
from mindspore._c_expression import get_code_extra
from tests.mark_utils import arg_mark

@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping tests on Python 3.11 and higher.")

def for_range(x):
    res = 0
    for i in range(x):
        res = res + i
    return res


def for_enumerate(x):
    x = [x, x, x]
    res = 0
    for i, v in enumerate(x):
        res = res + i
        res = res + v
    return x


def for_zip(x):
    x = [x, x, x]
    v = None
    for v in zip(x, x, x, x):
        pass
    return v


def for_mix(x):
    x = [x, x, x]
    res = 0
    for i, v in enumerate(list(zip(x, x, x, x))):
        res = res + i
        res = res + v[0]
    return res


def for_mix_with_sideeffect(x):
    x = [x, x, x]
    z = zip(list(enumerate(x)))
    for i in z:
        if i[0] == 1:
            break
    return list(z)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [for_range, for_enumerate, for_mix, for_zip])
@pytest.mark.parametrize('param', [1, Tensor([1])])
def test_for_iter_unrolling(func, param):
    """
    Feature: Test loop unrolling
    Description: Test loop unrolling
    Expectation: No exception.
    """
    if func is for_mix and param is not 1:
        pytest.skip(reason="fix later, Tensor parameter is constant and it's incorrect constant")

    config = {"loop_unrolling": True}
    excepted = func(param)
    result = jit(fn=func, mode="PIJit", jit_config=config)(param)
    jcr = get_code_extra(func)

    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["code"]["call_count_"] > 0
    assert excepted == result


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [for_mix_with_sideeffect])
@pytest.mark.parametrize('param', [1, Tensor([1])])
def test_not_implement_for_iter(func, param):
    """
    Feature: Test loop unrolling
    Description: Test loop unrolling
    Expectation: No exception.
    """
    config = {"loop_unrolling": True}
    excepted = func(param)
    result = jit(fn=func, mode="PIJit", jit_config=config)(param)
    jcr = get_code_extra(func)

    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["code"]["call_count_"] > 0
    assert excepted == result
