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
import os
import re
import shutil
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.runtime import Stream
from mindspore.runtime import StreamCtx as MsJitStreamCtx
from mindspore.ops.functional import grad
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


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_validateir_file(save_path)
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    clean_all_ir_files(save_path)
    return content


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


    save_path = "./test_my_ms_jit_stream_ctx"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == x.asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 1

    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

    save_path = "./test_my_ms_jit_stream_ctx_runtime"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == x.asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 1
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


    save_path = "./test_my_ms_jit_stream_ctx_mutli"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxMutliNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == (x * 2).asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


    save_path = "./test_my_ms_jit_stream_ctx_nest"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxMutliNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == (x * 2).asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


    save_path = "./test_basic_stream_block_annotation_1"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MsJitStreamNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    con = 0
    result = net(x, con)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(result, Tensor(np.ones([3, 3], dtype=np.float32)) * 23)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 3
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, con)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, con)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


    save_path = "./test_basic_stream_block_annotation_2"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = SimpleStreamNet()
    result = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 3
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

    save_path = "./test_nested_stream_blocks"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = NestedStreamNet()
    result = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 5)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 4
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

    save_path = "./test_complex_nested_streams"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = ComplexStreamsNet()
    result = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 6)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 5
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


    save_path = "./test_multiple_independent_streams"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([2, 2], dtype=np.float32))
    y = Tensor(np.ones([2, 2], dtype=np.float32))
    net = MutliStreamsNet()
    result = net(x, y)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 6)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 4
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, y)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, y)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    net = ConditionalStreamNet()
    result_with_stream = net(x, True)
    result_without_stream = net(x, False)
    assert np.allclose(result_with_stream, result_without_stream)
    assert np.allclose(result_with_stream, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, True)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, False)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@pytest.mark.skip(reason='Not support yet, UnboundLocalError')
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    net = StreamWithBreakNet()
    result = net(x, 5)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 7)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, 5)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, 5)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    net = StreamWithContinueNet()
    result = net(x, 4)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)))
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, 4)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, 4)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()



@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    net = StreamWithReturnNet()
    result_early = net(x, True)
    assert np.allclose(result_early, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)

    result_full = net(x, False)
    assert np.allclose(result_full, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)

    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, True)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, False)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    net = StreamWithReturnNet()
    result1 = net(x, 1)
    assert np.allclose(result1, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)

    result2 = net(x, 2)
    assert np.allclose(result2, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)

    result3 = net(x, 3)
    assert np.allclose(result3, Tensor(np.ones([2, 2], dtype=np.float32)) * 5)

    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, 1)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, 1)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()
