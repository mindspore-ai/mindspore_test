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
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
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


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_event_record():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """

    class WithEventNet(nn.Cell):
        def __init__(self):
            super(WithEventNet, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                event = self.depend(event, z)
                event.record()
                event.wait()
            return y + z

    ms.set_context(mode=ms.GRAPH_MODE)
    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet()
    out = net(x)
    print("out:", out)


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_event_no_return():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """

    class WithEventNet(nn.Cell):
        def construct(self, x):
            event = ms.runtime.Event()
            event.record()
            event.wait()
            return x

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet()
    out = net(x)
    print("out:", out)


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_event_record_multi_streams():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """

    class WithEventNet(nn.Cell):
        def __init__(self):
            super(WithEventNet, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                event = self.depend(event, z)
                event.record()
            with MyMsJitStreamCtx(s2):
                z1 = a + b + x
                event = self.depend(event, z1)
                event.wait()
                z = z + z1
            return y + z

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet()
    out = net(x)
    print("out:", out)


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_event_record_multi_events():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """

    class WithEventNet(nn.Cell):
        def __init__(self):
            super(WithEventNet, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event1 = ms.runtime.Event()
            event2 = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                event1.record()
                event1.wait()
                output = y + x
                event2.record()
            event2.wait()
            output = x - y * (output / 2)
            return output

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet()
    out = net(x)
    print("out:", out)


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_event_wait_before_record():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """

    class WithEventNet(nn.Cell):
        def __init__(self):
            super(WithEventNet, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            event = ms.runtime.Event()
            y = x * 2
            event = self.depend(event, y)
            event.wait()
            x = self.depend(x, event)
            z = a + b + x
            event = self.depend(event, z)
            event.record()
            z = self.depend(z, event)
            z = z + 1
            return y + z

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet()
    out = net(x)
    print("out:", out)
