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
""" test graph with while statement. """
import mindspore as ms
from mindspore.nn import Cell
from mindspore import Tensor
import numpy as np

ms.set_context(mode=ms.GRAPH_MODE)
def test_while_else_basic_without_half_unroll():
    """
    Feature: while else.
    Description: test while else.
    Expectation: No exception.
    """
    class WhileLoopBasic(Cell):
        def __init__(self):
            super().__init__()
            self.i = 3

        def construct(self, x):
            output = x
            while self.i > 0:
                output += 1
                self.i -= 1
            output += 4
            return output

    net = WhileLoopBasic()
    x = 1
    output = net(x)
    assert output == 1+1+1+1+4

def test_while_else_basic_without_half_unroll_with_break():
    """
    Feature: while else.
    Description: test while else.
    Expectation: No exception.
    """
    class WhileLoopBasicAndBreak(Cell):
        def __init__(self):
            super().__init__()
            self.i = 3

        def construct(self, x):
            output = x
            while self.i > 0:
                self.i -= 1
                if self.i == 2: break
                output += self.i
            else:
                output += Tensor(np.array(4).astype(np.int32))
            return output

    net = WhileLoopBasicAndBreak()
    x = Tensor(np.array(1).astype(np.int32))
    output = net(x)
    assert output.asnumpy() == 1

def test_while_else_basic_without_half_unroll_with_continue():
    """
    Feature: while else.
    Description: test while else.
    Expectation: No exception.
    """
    class WhileLoopBasicAndContinue(Cell):
        def __init__(self):
            super().__init__()
            self.i = 3

        def construct(self, x):
            output = x
            while self.i > 0:
                self.i -= 1
                if self.i == 2: continue
                output += 1
            output += Tensor(np.array(4).astype(np.int32))
            return output
    net = WhileLoopBasicAndContinue()
    x = Tensor(np.array(1).astype(np.int32))
    output = net(x)
    assert output.asnumpy() == 1+1+1+4

def test_while_else_basic_without_half_unroll_with_nested_while():
    """
    Feature: while else.
    Description: test while else.
    Expectation: No exception.
    """
    class WhileLoopBasicAndNestedWhile(Cell):
        def __init__(self):
            super().__init__()
            self.i = 3

        def construct(self, x):
            output = x
            x = self.i
            while x > 1:
                j = 1
                while j < x:
                    output += x*j
                    j += 1
                output += Tensor(np.array(4).astype(np.int32))
                x -= 1
            output += Tensor(np.array(4).astype(np.int32))
            return output

    net = WhileLoopBasicAndNestedWhile()
    x = Tensor(np.array(1).astype(np.int32))
    output = net(x)
    assert output.asnumpy() == 1+1*3+2*3+4+1*2+4+4
