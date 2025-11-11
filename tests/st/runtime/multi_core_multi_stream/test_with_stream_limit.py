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
"""Test with StreamCtx and res_limit"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={'jit_level': 'O0'})
ms.set_context(save_graphs=True, save_graphs_path="./ir")

a = Tensor(np.ones([3, 3]), ms.float32)
b = Tensor(np.ones([3, 3]), ms.float32)
s1 = ms.runtime.Stream()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_jit_stream_ctx():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with ms.runtime.StreamCtx(s1):
                z = a + b + x
            return z - y

    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    assert (out.asnumpy() == x.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_jit_stream_limit_ctx():
    """
    Feature: runtime stream api.
    Description: Test runtime.StreamLimitCtx api.
    Expectation: runtime.StreamLimitCtx api performs as expected.
    """
    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with ms.runtime.StreamLimitCtx(s1, 8, 8):
                z = a + b + x
            return z - y

    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    assert (out.asnumpy() == x.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_jit_stream_limit_ctx1():
    """
    Feature: runtime stream api.
    Description: Test runtime.StreamLimitCtx api.
    Expectation: runtime.StreamLimitCtx api performs as expected.
    """
    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with ms.runtime.StreamCtx(s1):
                with ms.runtime.StreamLimitCtx(s1, 8, 8):
                    z = a + b + x
            return z - y

    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    assert (out.asnumpy() == x.asnumpy()).all()
