# Copyright 2021 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import mindspore as ms
from tests.mark_utils import arg_mark


class RLBufferAppend(nn.Cell):
    def __init__(self, capacity, shapes, types):
        super(RLBufferAppend, self).__init__()
        self._capacity = capacity
        self.count = Parameter(Tensor(0, ms.int32), name="count")
        self.head = Parameter(Tensor(0, ms.int32), name="head")
        self.buffer_append = P.BufferAppend(self._capacity, shapes, types)

    @jit
    def construct(self, buffer, exps):
        return self.buffer_append(buffer, exps, self.count, self.head)


class RLBufferGet(nn.Cell):
    def __init__(self, capacity, shapes, types):
        super(RLBufferGet, self).__init__()
        self._capacity = capacity
        self.count = Parameter(Tensor(5, ms.int32), name="count")
        self.head = Parameter(Tensor(0, ms.int32), name="head")
        self.buffer_get = P.BufferGetItem(self._capacity, shapes, types)

    @jit
    def construct(self, buffer, index):
        return self.buffer_get(buffer, self.count, self.head, index)


class RLBufferSample(nn.Cell):
    def __init__(self, capacity, batch_size, shapes, types):
        super(RLBufferSample, self).__init__()
        self._capacity = capacity
        self.count = Parameter(Tensor(5, ms.int32), name="count")
        self.head = Parameter(Tensor(0, ms.int32), name="head")
        self.buffer_sample = P.BufferSample(self._capacity, batch_size, shapes, types)

    @jit
    def construct(self, buffer):
        return self.buffer_sample(buffer, self.count, self.head)


states = Tensor(np.arange(4 * 5).reshape(5, 4).astype(np.float32) / 10.0)
actions = Tensor(np.arange(2 * 5).reshape(5, 2).astype(np.int32))
rewards = Tensor(np.ones((5, 1)).astype(np.int32))
states_ = Tensor(np.arange(4 * 5).reshape(5, 4).astype(np.float32))
b = [states, actions, rewards, states_]

s = Tensor(np.array([2, 2, 2, 2]), ms.float32)
a = Tensor(np.array([0, 0]), ms.int32)
r = Tensor(np.array([0]), ms.int32)
s_ = Tensor(np.array([3, 3, 3, 3]), ms.float32)
exp = [s, a, r, s_]
exp1 = [s_, a, r, s]

c = [Tensor(np.array([[6, 6, 6, 6], [6, 6, 6, 6]]), ms.float32),
     Tensor(np.array([[6, 6], [6, 6]]), ms.int32),
     Tensor(np.array([[6], [6]]), ms.int32),
     Tensor(np.array([[6, 6, 6, 6], [6, 6, 6, 6]]), ms.float32)]


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_BufferSample():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    buffer_sample = RLBufferSample(capacity=5, batch_size=3, shapes=[(4,), (2,), (1,), (4,)], types=[
        ms.float32, ms.int32, ms.int32, ms.float32])
    ss, aa, rr, ss_ = buffer_sample(b)
    print(ss, aa, rr, ss_)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_BufferGet():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    buffer_get = RLBufferGet(capacity=5, shapes=[(4,), (2,), (1,), (4,)], types=[
        ms.float32, ms.int32, ms.int32, ms.float32])
    ss, aa, rr, ss_ = buffer_get(b, 1)
    expect_s = [0.4, 0.5, 0.6, 0.7]
    expect_a = [2, 3]
    expect_r = [1]
    expect_s_ = [4, 5, 6, 7]
    np.testing.assert_almost_equal(ss.asnumpy(), expect_s)
    np.testing.assert_almost_equal(aa.asnumpy(), expect_a)
    np.testing.assert_almost_equal(rr.asnumpy(), expect_r)
    np.testing.assert_almost_equal(ss_.asnumpy(), expect_s_)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_BufferAppend():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    buffer_append = RLBufferAppend(capacity=5, shapes=[(4,), (2,), (1,), (4,)], types=[
        ms.float32, ms.int32, ms.int32, ms.float32])

    buffer_append(b, exp)
    buffer_append(b, exp)
    buffer_append(b, exp)
    buffer_append(b, exp)
    buffer_append(b, exp)
    buffer_append(b, exp1)
    expect_s = [[3, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
    expect_a = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    expect_r = [[0], [0], [0], [0], [0]]
    expect_s_ = [[2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
    np.testing.assert_almost_equal(b[0].asnumpy(), expect_s)
    np.testing.assert_almost_equal(b[1].asnumpy(), expect_a)
    np.testing.assert_almost_equal(b[2].asnumpy(), expect_r)
    np.testing.assert_almost_equal(b[3].asnumpy(), expect_s_)
