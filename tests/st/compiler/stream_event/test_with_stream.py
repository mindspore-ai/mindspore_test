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
"""Test with stream"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn
from mindspore.runtime.ms_jit_stream_ctx import MsJitStream, MsJitStreamCtx
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


class MyMsJitStreamCtx(MsJitStreamCtx):
    def __init__(self, ctx_stream):
        self.stream = ctx_stream
        self.prev_stream = None

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


a = Tensor(np.ones([3, 3]), ms.float32)
b = Tensor(np.ones([3, 3]), ms.float32)
s1 = MsJitStream()
s2 = MsJitStream()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_my_ms_jit_stream_ctx():
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
            return z - y

    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    assert (out.asnumpy() == x.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_my_ms_jit_stream_ctx_mutli():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MyMsJitStreamCtxMutliNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
            with MyMsJitStreamCtx(s2):
                y = a - y
            return y + z

    net = MyMsJitStreamCtxMutliNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    assert(out.asnumpy() == (x * 2).asnumpy()).all()


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_my_ms_jit_stream_ctx_nest():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MyMsJitStreamCtxMutliNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                with MyMsJitStreamCtx(s2):
                    y = a - y
            return y + z

    net = MyMsJitStreamCtxMutliNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    print("x:", x)
    out = net(x)
    print("out:", out)
