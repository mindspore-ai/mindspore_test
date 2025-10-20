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

    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(out.asnumpy(), Tensor(np.zeros([3, 3]), ms.float32).asnumpy(), 1e-3, 1e-3)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2
    assert len(vector_num) == 1
    assert len(cube_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_limit_runtime():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            with ms.runtime.StreamCtx(s1):
                y = x * 2
                with ms.runtime.StreamLimitCtx(s1, 8, 8):
                    z = a + b + x
            return z - y


    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit_runtime"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(out.asnumpy(), x.asnumpy(), 1e-3, 1e-3)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2
    assert len(vector_num) == 1
    assert len(cube_num) == 1



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

    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit_diff_stream"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(out.asnumpy(), Tensor(np.zeros([3, 3]), ms.float32).asnumpy(), 1e-3, 1e-3)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2
    assert not vector_num
    assert not cube_num
