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
import pytest

import mindspore
from mindspore import JitConfig, Profiler

from tests.mark_utils import arg_mark


class DynamicQuantCell(mindspore.nn.Cell):
    def __init__(self):
        super(DynamicQuantCell, self).__init__()
        self.quant = mindspore.ops.auto_generate.DynamicQuantExt()

    def construct(self, x, smooth_scales=None):
        return self.quant(x, smooth_scales)


def get_expect(x, smooth_scales=None):
    if smooth_scales is not None:
        x = x * smooth_scales
    x_abs = np.abs(x)
    x_max = x_abs.max(axis=-1, keepdims=True).astype(np.float32)
    scale = x_max / 127.0
    x = x.astype(np.float32) / scale
    output = np.round(x).astype(np.int8)
    scale = np.squeeze(scale, axis=-1)

    return output, scale


def random_input(shape, dtype=mindspore.float16):
    return mindspore.Tensor(np.random.randn(*shape), dtype=dtype)


def run_dynamic_quant(dtype, mode="KBK", batch=1, seq=1, hidden=512):
    quant = DynamicQuantCell()
    if mode == 'pynative':
        mindspore.context.set_context(mode=mindspore.PYNATIVE_MODE)
    elif mode == 'GE':
        mindspore.context.set_context(mode=mindspore.GRAPH_MODE)
    elif mode == 'KBK':
        mindspore.context.set_context(mode=mindspore.GRAPH_MODE)
        quant.set_jit_config(JitConfig(jit_level='O0', infer_boost=on"))
    x = random_input((batch, seq, hidden), dtype)
    x_np = x.float().asnumpy()
    output, scale = quant(x)
    output_expect, scale_expect = get_expect(x_np)
    np.testing.assert_allclose(output.asnumpy(), output_expect, atol=1)
    np.testing.assert_allclose(scale.asnumpy(), scale_expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend310p', 'platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('batch', [1])
@pytest.mark.parametrize('hidden', [512, 1024])
def test_dynamic_quant_f16_small(batch, hidden):
    """
    Feature:ops.DynamicQuantExt use internal op
    Description: ops.DynamicQuantExt with small shape
    Expectation: Success
    """
    run_dynamic_quant(mindspore.float16, batch=batch, hidden=hidden)


@arg_mark(plat_marks=['platform_ascend310p', 'platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('batch', [32])
@pytest.mark.parametrize('hidden', [4608, 7168])
def test_dynamic_quant_f16_large(batch, hidden):
    """
    Feature:ops.DynamicQuantExt use internal op
    Description: ops.DynamicQuantExt with large shape
    Expectation: Success
    """
    run_dynamic_quant(mindspore.float16, batch=batch, hidden=hidden)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_quant_bf16(mode):
    """
    Feature:ops.DynamicQuantExt use internal op
    Description: ops.DynamicQuantExt with bfloat16
    Expectation: Success
    """
    run_dynamic_quant(mindspore.bfloat16, mode)
