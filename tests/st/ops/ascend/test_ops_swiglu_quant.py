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
""" test SwigluQuant op """
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, Parameter
from mindspore import ops

class SwigluQuantNet(nn.Cell):
    """Swiglu + QuantV2"""
    def __init__(self, shape, dim):
        super(SwigluQuantNet, self).__init__()
        self.silu = nn.SiLU()
        self.mul = ms.ops.Mul()
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.quant = ops.auto_generate.QuantV2()
        self.dim = dim
        self.shape = shape

    def construct(self, x, smooth_scales=None, offset=None):
        shp = self.shape[self.dim] // 2
        sizes = [shp, shp]
        gate, hidden = self.split_with_size(x, sizes, self.dim)
        gate = self.silu(gate)
        res = self.mul(hidden, gate)
        hidden_states = self.quant(res, smooth_scales, offset)
        return hidden_states

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gen_np_output(output):
    if output.dtype == ms.bfloat16:
        output_np = output.float().asnumpy()
    else:
        output_np = output.asnumpy()
    return output_np

def static_per_channel_quant(swiglu_out, scale, offset, x_row_len):
    scale = np.reshape(scale, [1, -1])
    offset = np.reshape(offset, [1, -1])
    group_index = np.random.choice(range(1, x_row_len - 1), 0, replace=False).astype(np.int32)
    group_index = np.sort(group_index)
    group_index = np.append(group_index, x_row_len).astype(np.int32)

    y_tmp = swiglu_out.copy()
    y_tmp[0: group_index[0], :] = swiglu_out[0: group_index[0], :] * scale[0, :]
    y_tmp[0: group_index[0], :] = y_tmp[0: group_index[0], :] + offset[0, :]
    y_tmp = np.round(y_tmp)
    y_tmp = np.clip(y_tmp, -128, 127)
    quant_out = y_tmp.astype(np.int8)
    return quant_out

def get_expect(x, dim, np_dtype, smooth_scales=None, offset=None):
    x0, x1 = np.split(x, 2, axis=dim)
    x0_cast = x0.astype(np_dtype)
    x1_cast = x1.astype(np_dtype)
    swiglu_res = sigmoid(x0_cast) * x0_cast
    swiglu_res *= x1_cast

    return static_per_channel_quant(swiglu_res, smooth_scales, offset, x.shape[0])

def _test_swiglu_quant(shape, dim, dtype, np_dtype, is_dyn=False, prof=False):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    context.set_context(save_graphs=True, save_graphs_path="./swiglu_Quant_graph")

    x_np = np.random.normal(-2, 2, size=shape).astype(np_dtype)
    t_shape = shape[dim] // 2
    smooth_scales_np = np.random.randn(t_shape).astype(np_dtype)
    offset_np = np.random.randn(t_shape).astype(np_dtype)
    input_data = Tensor(x_np, dtype=dtype)
    input_scale = Parameter(Tensor(smooth_scales_np, dtype=ms.float16), name="scale")
    input_offset = Parameter(Tensor(offset_np, dtype=ms.float32), name="offset")
    net = SwigluQuantNet(shape, dim)

    if is_dyn:
        input_dyn = Tensor(shape=[None] * len(shape), dtype=dtype)
        net.set_inputs(input_dyn, input_scale, input_offset)
    if prof:
        for _ in range(50):
            output = net(input_data, input_scale, input_offset)
        return
    output = net(input_data, input_scale, input_offset)
    output_expect = get_expect(x_np, dim, np_dtype, smooth_scales_np, offset_np)
    output_np = gen_np_output(output)

    np.testing.assert_allclose(output_np, output_expect, atol=1, rtol=1e-2)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('shape', [(16, 6144), (256, 5502), (512, 6700)])
@pytest.mark.parametrize('dim', [-1])
@pytest.mark.parametrize('np_dtype', [np.float32])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_swiglu_quant_bfloat16(shape, dim, dtype, np_dtype, is_dyn):
    """
    Feature: Test SwigluQuant.
    Description: Test SwigluQuant fusion.
    Expectation: Success.
    """
    _test_swiglu_quant(shape, dim, dtype, np_dtype, is_dyn)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('shape', [(16, 6144), (256, 6900)])
@pytest.mark.parametrize('dim', [1])
@pytest.mark.parametrize('np_dtype', [np.float32])
@pytest.mark.parametrize('dtype', [ms.float16])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_swiglu_quant_float16(shape, dim, dtype, np_dtype, is_dyn):
    """
    Feature: Test SwigluQuant.
    Description: Test SwigluQuant fusion.
    Expectation: Success.
    """
    _test_swiglu_quant(shape, dim, dtype, np_dtype, is_dyn)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('shape', [(2, 2152), (16, 7168), (4096, 8192), (16, 6868)])
@pytest.mark.env_onecard
def test_swiglu_quant_dyn_shape(shape):
    """
    Feature: Test SwigluQuant.
    Description: Test SwigluQuant fusion.
    Expectation: Success.
    """
    _test_swiglu_quant(shape, dim=-1, dtype=ms.bfloat16, np_dtype=np.float32, is_dyn=True)
