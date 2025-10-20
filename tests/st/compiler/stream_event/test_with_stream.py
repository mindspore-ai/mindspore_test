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
from mindspore.runtime import Stream
from mindspore.runtime import StreamCtx as MsJitStreamCtx
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={'jit_level': 'O0'})


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
s1 = Stream()
s2 = Stream()
s3 = Stream()


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
def test_my_ms_jit_stream_ctx_runtime():
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
    assert (out.asnumpy() == (x * 2).asnumpy()).all()


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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_basic_stream_block_annotation_1():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MsJitStreamNet(nn.Cell):
        def construct(self, x, con):
            y = x * 2
            with MsJitStreamCtx(s1):
                z = a + x
                if con < 5:
                    z = z + 5
                y = y + z
            y = y + z
            return y + z

    x = Tensor(np.ones([3, 3]), ms.float32)
    con = 0
    result = MsJitStreamNet()(x, con)
    assert np.allclose(result, Tensor(np.ones([3, 3], dtype=np.float32)) * 23)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_basic_stream_block_annotation_2():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class SimpleStreamNet(nn.Cell):
        def construct(self, x):
            x = ops.abs(x)
            with MsJitStreamCtx(s1):
                x = x + Tensor(np.ones([2, 2], dtype=np.float32))
                x = ops.relu(x)
            x = ops.abs(x)
            with MsJitStreamCtx(s2):
                x = x + Tensor(np.ones([2, 2], dtype=np.float32))
            x = ops.abs(x)
            return x

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    result = SimpleStreamNet()(x)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_stream_blocks():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class NestedStreamNet(nn.Cell):
        def construct(self, x):
            with MsJitStreamCtx(s1):
                x1 = x + Tensor(np.ones([2, 2], dtype=np.float32))
                with MsJitStreamCtx(s2):
                    x2 = x1 + Tensor(np.ones([2, 2], dtype=np.float32))
                    x3 = x2 + Tensor(np.ones([2, 2], dtype=np.float32))
                x4 = x3 + Tensor(np.ones([2, 2], dtype=np.float32))
            return x4

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    result = NestedStreamNet()(x)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_complex_nested_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class ComplexStreamsNet(nn.Cell):
        def construct(self, x):
            with MsJitStreamCtx(s1):
                t = Tensor(np.ones([2, 2], dtype=np.float32))
                x1 = x + t
                with MsJitStreamCtx(s2):
                    x2 = x1 + t
                    with MsJitStreamCtx(s3):
                        x3 = x2 + t
                    x4 = x3 + t
                x5 = x4 + t
            return x5

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    result = ComplexStreamsNet()(x)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 6)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multiple_independent_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MutliStreamsNet(nn.Cell):
        def construct(self, x, y):
            with MsJitStreamCtx(s1):
                x1 = x + Tensor(np.ones([2, 2], dtype=np.float32))
                x_relu = ops.relu(x1)

            intermediate = x + y

            with MsJitStreamCtx(s2):
                y1 = y + Tensor(np.ones([2, 2], dtype=np.float32))
                y_relu = ops.relu(y1)

            result = x_relu + y_relu
            result = result + intermediate
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    y = Tensor(np.ones([2, 2], dtype=np.float32))
    result = MutliStreamsNet()(x, y)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 6)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_stream_with_control_flow():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class ConditionalStreamNet(nn.Cell):
        def construct(self, x, use_stream):
            if use_stream:
                with MsJitStreamCtx(s1):
                    x = x + Tensor(np.ones([2, 2], dtype=np.float32))
            else:
                x = x + Tensor(np.ones([2, 2], dtype=np.float32))
            return x

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    result_with_stream = ConditionalStreamNet()(x, True)
    result_without_stream = ConditionalStreamNet()(x, False)
    assert np.allclose(result_with_stream, result_without_stream)
    assert np.allclose(result_with_stream, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)


@pytest.mark.skip(reason='Not support yet, UnboundLocalError')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_switch_like_control_flow_with_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class ConditionalStreamNet(nn.Cell):
        def construct(self, x, condition):
            result = x * 2
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            if condition == 0:
                with MsJitStreamCtx(s1):
                    result = result + t
                    return result
            elif condition == 1:
                with MsJitStreamCtx(s2):
                    result = result + t * 3
                result = result + t
            elif condition == 2:
                result = result + t * 5
                return result
            else:
                result = result - t
                with MsJitStreamCtx(s3):
                    result = result + t * 7
            result = ops.relu(result)
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))

    result0 = ConditionalStreamNet()(x, 0)
    assert np.allclose(result0, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)

    result1 = ConditionalStreamNet()(x, 1)
    assert np.allclose(result1, Tensor(np.ones([2, 2], dtype=np.float32)) * 6)

    result2 = ConditionalStreamNet()(x, 2)
    assert np.allclose(result2, Tensor(np.ones([2, 2], dtype=np.float32)) * 7)

    result3 = ConditionalStreamNet()(x, 3)
    assert np.allclose(result3, Tensor(np.ones([2, 2], dtype=np.float32)) * 8)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_stream_with_break_statement():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class StreamWithBreakNet(nn.Cell):
        def construct(self, x, max_iter):
            result = x
            i = 0
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            while i < max_iter:
                with MsJitStreamCtx(s1):
                    result = result + t
                    if i >= 2:
                        result = result - t
                        break
                    result = result + t * 2
                i += 1
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    result = StreamWithBreakNet()(x, 5)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 7)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_stream_with_continue_statement():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class StreamWithContinueNet(nn.Cell):
        def construct(self, x, max_iter):
            result = x
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            i = 0
            while i < max_iter:
                with MsJitStreamCtx(s1):
                    if i % 2 == 0:
                        result = result - t
                        i += 1
                        continue
                    result = result + t
                i += 1
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    result = StreamWithContinueNet()(x, 4)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_stream_with_return_statement():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class StreamWithReturnNet(nn.Cell):
        def construct(self, x, early_return):
            x = ops.abs(x)
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            with MsJitStreamCtx(s1):
                x1 = x + t
                if early_return:
                    return x1
                relu1 = ops.relu(x1)
            with MsJitStreamCtx(s2):
                x2 = relu1 + t
                result = ops.relu(x2)
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))

    result_early = StreamWithReturnNet()(x, True)
    assert np.allclose(result_early, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)

    result_full = StreamWithReturnNet()(x, False)
    assert np.allclose(result_full, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multiple_returns_in_different_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class StreamWithReturnNet(nn.Cell):
        def construct(self, x, condition):
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            with MsJitStreamCtx(s1):
                x1 = x + t
                if condition == 1:
                    return x1
            with MsJitStreamCtx(s2):
                x2 = x + t * 2
                if condition == 2:
                    return x2
            with MsJitStreamCtx(s3):
                x3 = x1 + x2
                return x3

    x = Tensor(np.ones([2, 2], dtype=np.float32))

    result1 = StreamWithReturnNet()(x, 1)
    assert np.allclose(result1, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)

    result2 = StreamWithReturnNet()(x, 2)
    assert np.allclose(result2, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)

    result3 = StreamWithReturnNet()(x, 3)
    assert np.allclose(result3, Tensor(np.ones([2, 2], dtype=np.float32)) * 5)
