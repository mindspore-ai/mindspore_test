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
import mindspore as ms
import mindspore.nn as nn
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_event():
    """
    Feature: Support event in graph mode.
    Description: Support event in graph mode.
    Expectation: Run success.
    """

    class EventNet(nn.Cell):
        def construct(self, x):
            return x

    event1 = ms.runtime.Event()
    net = EventNet()
    out = net(event1)
    print("out:", out)


@pytest.mark.skip(reason='Not support yet')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_event_record_wait():
    """
    Feature: Support event.record() and event.wait() in graph mode.
    Description: Support event.record() and event.wait() in graph mode.
    Expectation: Run success.
    """

    class EventNet(nn.Cell):
        def construct(self, x):
            x.record()
            return x.wait()

    event1 = ms.runtime.Event()
    net = EventNet()
    out = net(event1)
    print("out:", out)
