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
#from tests.mark_utils import arg_mark
import numpy as np
import pytest
import torch
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, ops
from tests.mark_utils import arg_mark

class DequantSwigluQuantNet(nn.Cell):
    def __init__(self):
        super(DequantSwigluQuantNet, self).__init__()
        self.dequant_swiglu_quant = ops.auto_generate.DequantSwigluQuant()

    def construct(self, x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index,
                  activate_left=False, quant_mode="static"):
        y_out, scale_out = self.dequant_swiglu_quant(
            x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index, activate_left,
            quant_mode)
        return y_out, scale_out

def golden_dequant_swiglu_quant_torch(
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_num,
        activate_left,
        quant_mode,
    ):
    x = x.to(torch.float32)
    weight_scale = weight_scale.to(torch.float32)
    activation_scale = activation_scale.to(torch.float32)
    res = torch.mul(x, weight_scale)
    res = torch.mul(res, activation_scale)
    out = torch.chunk(res, 2, dim=-1)
    if activate_left:
        self_tensor = out[0]
        other = out[1]
    else:
        self_tensor = out[1]
        other = out[0]
    output = torch.nn.functional.silu(self_tensor) * other
    if quant_scale is not None:
        output = torch.mul(output, quant_scale)
    scale_out = torch.zeros([x.shape[0]], dtype=torch.float32)
    if quant_mode == "dynamic":
        absolute = torch.abs(output)
        max_values = torch.amax(absolute, dim=-1)
        scale_out = max_values / 127.0
        max_values = 127.0 / max_values
        output = output * max_values.unsqueeze(1)
    output = torch.clamp(output, -128, 127)
    output = torch.round(output)
    return output.to(torch.int8).cpu().numpy(), scale_out.cpu().numpy()

def custom_compare(output, expect, mstype):
    if mstype == ms.float16:
        limit = 0.004
    elif mstype == ms.bfloat16:
        limit = 0.03
    elif mstype == ms.float32:
        limit = 0.004

    print("limit = ", limit)
    out_flatten = output.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum(np.abs(out_flatten - expect_flatten) /
                     np.abs(expect_flatten) > limit).astype(np.int32)
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        print("[FAILED]", "err_cnt = ", err_cnt, "/", limit_cnt)
        return False

    print("[SUCCESS]", "err_cnt = ", err_cnt, "/", limit_cnt)
    return True

def DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode):
    bias_tensor = None if bias is None else Tensor(bias)
    quant_scale_tensor = None if quant_scale is None else Tensor(quant_scale)
    quant_offset_tensor = None if quant_offset is None else Tensor(quant_offset)
    group_index_tensor = None if group_index is None else Tensor(group_index)
    y_out, scale_out = net(
        Tensor(x),
        Tensor(weight_scale),
        Tensor(activation_scale),
        bias_tensor,
        quant_scale_tensor,
        quant_offset_tensor,
        group_index_tensor,
        activate_left,
        quant_mode)

    x_torch = torch.from_numpy(x)
    weight_scale_torch = torch.from_numpy(weight_scale)
    activatition_scale_torch = torch.from_numpy(activation_scale)
    bias_torch = None if bias is None else torch.from_numpy(bias)
    quant_scale_torch = None if quant_scale is None else torch.from_numpy(quant_scale)
    quant_offset_torch = None if quant_offset is None else torch.from_numpy(quant_offset)
    group_index_torch = None if group_index is None else torch.from_numpy(group_index)
    y_out_golden, scale_out_golden = golden_dequant_swiglu_quant_torch(
        x_torch,
        weight_scale_torch,
        activatition_scale_torch,
        bias_torch,
        quant_scale_torch,
        quant_offset_torch,
        group_index_torch,
        activate_left,
        quant_mode)

    y_diff = y_out.asnumpy().flatten() - y_out_golden.flatten()
    y_max_diff = np.max(np.abs(y_diff))
    y_compare_result = y_max_diff <= 1

    scale_compare_result = custom_compare(scale_out.asnumpy(), scale_out_golden, ms.float32)
    assert y_compare_result and scale_compare_result, "dequant_swiglu_quant compare failed"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dequant_swiglu_quant_non_group_index(context_mode):
    '''
    Feature:aclnnDequantSwigluQuant kernel.
    Description: test for aclnnDequantSwigluQuant ops.
    Expectation:should pass for all testcases.
    '''
    ms.set_context(device_target="Ascend", mode=context_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # ms.set_context(save_graphs=True, save_graphs_path="./dequant_swiglu_quant_ir")

    tokensNum = 4608
    H = 1024
    x = np.random.randint(-10, 10, size=(tokensNum, 2 * H), dtype=np.int32)
    weight_scale = np.random.randn(2 * H).astype(np.float32)
    activation_scale = np.random.randn(tokensNum, 1).astype(np.float32)
    bias = None
    quant_scale = np.random.randn(1, H).astype(np.float32)
    quant_offset = None
    group_index = None
    activate_left = False
    quant_mode = "dynamic"

    net = DequantSwigluQuantNet()
    DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('param', [[32, 2048], [128, 4608]])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dequant_swiglu_quant_group_x_weight_activation(param, context_mode):
    '''
    Feature:aclnnDequantSwigluQuant kernel.
    Description: test for aclnnDequantSwigluQuant ops.
    Expectation:should pass for all testcases.
    '''
    ms.set_context(device_target="Ascend", mode=context_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # ms.set_context(save_graphs=True, save_graphs_path="./dequant_swiglu_quant_ir")

    tokensNum, H = param
    x = np.random.randint(-10, 10, size=(tokensNum, 2 * H), dtype=np.int32)
    weight_scale = np.random.randn(2 * H).astype(np.float32)
    activation_scale = np.random.randn(tokensNum, 1).astype(np.float32)
    bias = None
    quant_scale = None
    quant_offset = None
    group_index = None
    activate_left = False
    quant_mode = "static"

    net = DequantSwigluQuantNet()
    DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode)
