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

import os
import numpy as np
import pytest

import mindspore as ms
from mindspore import context
from mindspore.ops.auto_generate import QuantBatchMatmul


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)


def np_qbmm_compute(a, b, tmp, bias=None):
    c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.int32)
    if bias is not None:
        c = c + bias
    c = c.astype(np.float32) * tmp
    c = c.astype(np.float16)
    return c


class Qbmm(ms.nn.Cell):
    def __init__(self, weight, scale, bias, ta, tb, dst_dtype):
        super().__init__()
        self.dbmm = QuantBatchMatmul(transpose_x1=ta,
                                     transpose_x2=tb,
                                     dtype=dst_dtype)
        self.weight = ms.Parameter(weight, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.bias = ms.Parameter(bias, requires_grad=False)

    def construct(self, x):
        return self.dbmm(x, self.weight, self.scale, None, self.bias)


class QbmmAdd(ms.nn.Cell):
    def __init__(self, weight, scale, bias, ta, tb, dst_dtype):
        super().__init__()
        self.dbmm = QuantBatchMatmul(transpose_x1=ta,
                                     transpose_x2=tb,
                                     dtype=dst_dtype)
        self.add = ms.ops.Add()
        self.weight = ms.Parameter(weight, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.bias = ms.Parameter(bias, requires_grad=False)

    def construct(self, x):
        return self.add(self.dbmm(x, self.weight, self.scale, None, None), self.bias)


def qbmm(m, k, n, trans_a=False, trans_b=False, with_outer_add=False, is_dyn=False):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    # NOTE: 在多卡并行的场景里，要控制随机种子以防多卡numpy生成不一致
    seed = 0
    np.random.seed(seed)
    a = np.random.uniform(-20, 20, size=(m, k)).astype(np.int8)
    np.random.seed(seed)
    b = np.random.uniform(-20, 20, size=(k, n)).astype(np.int8)
    np.random.seed(seed)
    bias = np.random.randint(-10, 10, (n)).astype(np.int32)
    np.random.seed(seed)
    tmp = np.random.rand(n).astype(np.float32) / 1000
    scale = process_deq_scale(tmp)
    bias_fp16 = bias * tmp
    expect_np = np_qbmm_compute(a, b, tmp, bias)

    if trans_a:
        a = np.transpose(a, (1, 0))
    if trans_b:
        b = np.transpose(b, (1, 0))

    a_ms = ms.Tensor(a, ms.int8)
    b_ms = ms.Tensor(b, ms.int8)

    scale_ms = ms.Tensor(scale, ms.int64)
    net = None
    if not with_outer_add:
        bias_ms = ms.Tensor(bias, ms.int32)
        net = Qbmm(b_ms, scale_ms, bias_ms, trans_a, trans_b, dst_dtype=ms.float16)
    else:
        bias_ms = ms.Tensor(bias_fp16, ms.float16)
        net = QbmmAdd(b_ms, scale_ms, bias_ms, trans_a, trans_b, dst_dtype=ms.float16)

    if is_dyn:
        input_dyn = ms.Tensor(shape=(None, None), dtype=ms.int8)
        net.set_inputs(input_dyn)

    output = net(a_ms)
    output_np = output.asnumpy()
    np.testing.assert_allclose(output_np, expect_np, rtol=0.002, atol=0.002)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('input_shape', [(128, 2560, 5120),
                                         (1024, 1024, 1024)])
def test_qbmm_False_True(input_shape):
    """
    Feature: testqbmm operator in graph mode
    Description: testqbmm.
    Expectation: the result is correct
    """
    m, k, n = input_shape
    qbmm(m, k, n, trans_a=False, trans_b=True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('input_shape', [(128, 512, 1024),
                                         (1024, 1024, 1024)])
@pytest.mark.parametrize('tb', [False, True])
@pytest.mark.parametrize('is_dyn', [False, True])
def test_qbmm_add(input_shape, tb, is_dyn):
    """
    Feature: test qbmm operator in graph mode
    Description: test qbmm.
    Expectation: the result is correct
    """
    m, k, n = input_shape
    qbmm(m, k, n, trans_a=False, trans_b=tb, with_outer_add=True, is_dyn=is_dyn)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qbmm_add_continuous():
    """
    Feature: test qbmm operator in graph mode
    Description: test qbmm.
    Expectation: the result is correct
    """
    qbmm(128, 128, 256, trans_a=False, trans_b=True, with_outer_add=True, is_dyn=False)
    qbmm(128, 64, 256, trans_a=False, trans_b=True, with_outer_add=True, is_dyn=False)
    qbmm(1280, 128, 256, trans_a=False, trans_b=False, with_outer_add=True, is_dyn=True)
