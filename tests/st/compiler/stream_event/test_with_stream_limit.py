# Copyright 2025 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor, ops
from mindspore.runtime import Stream, StreamCtx, StreamLimitCtx
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)

class MyMsJitStreamCtx(StreamCtx):
    def __init__(self, ctx_stream):
        self.stream = ctx_stream
        self.prev_stream = None

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


a = Tensor(np.ones([3, 3]), ms.float32)
b = Tensor(np.ones([3, 3]), ms.float32)

s1 = Stream()
s2 = Stream()

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_limit():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                with StreamLimitCtx(s1, 8, 8):
                    y = y + ops.abs(a)
            return z - y

    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    net(x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_limit_diff_stream():
    """
    Feature: Support with stream limit context.
    Description: Support with stream limit context.
    Expectation: Run success.
    """

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                with StreamLimitCtx(s2, 8, 8):
                    y = y + ops.abs(a)
            return z - y

    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    net(x)
