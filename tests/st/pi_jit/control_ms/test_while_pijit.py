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
"""test while in PIJit and pynative mode"""
from mindspore import Tensor, jit, context
from mindspore.common import dtype as mstype
import sys  
import pytest 
from tests.mark_utils import arg_mark

@pytest.fixture(autouse=True)  
def skip_if_python_version_too_high():  
    if sys.version_info >= (3, 11):  
        pytest.skip("Skipping tests on Python 3.11 and higher.") 

#@jit(mode="PIJit")
#TODO: fix CODEHOOK
@jit
def t1_while(x, y):
    y = y + 4
    while x < y:
        x = x + 1
    x = x + 3
    return x


#@jit(mode="PIJit")
#TODO: fix CODEHOOK
@jit
def const_branch(y):
    if y >= 0:
        while y > 1:
            y -= 1
        return y
    return 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_const_branch():
    """
    Feature: control flow .
    Description: Set one branch abstract with the other branch type
    when all the branches can not be inferred.
    Expectation: No error raised.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    y = Tensor(5)
    with pytest.raises(TypeError) as exc:
        with const_branch(y):
            pass

    assert "join" in str(exc.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net():
    """
    Feature: control flow .
    Description: Set one branch abstract with the other branch type
    when all the branches can not be inferred.
    Expectation: No error raised.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    c1 = Tensor([2], mstype.int32)
    c2 = Tensor([14], mstype.int32)
    expect = Tensor([21], mstype.int32)
    ret = t1_while(c1, c2)
    assert ret == expect


if __name__ == "__main__":
    test_net()
