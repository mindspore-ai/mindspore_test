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

import numpy as np

from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import context
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops.operations._infer_ops import QuantV2

class MoeInitRoutingQuantV2Net(Cell):
    def __init__(self):
        super().__init__()
        self.net = ops.auto_generate.MoeInitRoutingQuantV2()

    def construct(self, x, expert_idx, active_num, expert_capacity, expert_num,
                  drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag,
                  quant_mode, scale, offset):
        return self.net(x, expert_idx, active_num, expert_capacity, expert_num,
                        drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag,
                        quant_mode, scale, offset)

class MoeInitRoutingV2Net(Cell):
    def __init__(self):
        super().__init__()
        self.moe_init_routing_v2 = ops.auto_generate.MoeInitRoutingV2()
        self.quantV2 = QuantV2()

    def construct(self, x, expert_idx, active_num, expert_capacity, expert_num,
                  drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag,
                  quant_mode, scale, offset):
        moe_result = self.moe_init_routing_v2(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                                              expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag)
        print("---------- moe_result --------", moe_result[0])
        quant_out = self.quantV2(
            moe_result[0],
            scale,
            offset,
            False,
            "ROUND",
            ms.int8)
        return quant_out

def moe_init_routing_v2_test():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    context.set_context(
        save_graphs=1,
        save_graphs_path="./moe_init_routing_quantV2")

    num_rows = 100
    h = 256
    k = 20
    expert_num = 20
    active_num = 0
    expert_capacity = 50
    drop_pad_mode = 0
    expert_tokens_count_or_cumsum_flag = 0
    expert_tokens_before_capacity_flag = False
    max_expert_num = 1024

    x = ms.Tensor(np.random.uniform(-1, 1, size=(num_rows, h)
                                    ).astype(np.float32), ms.float32)

    if drop_pad_mode == 1 or (
            drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag > 0):
        expert_idx = ms.Tensor(
            np.random.randint(
                0, expert_num, size=(
                    num_rows, k)).astype(
                        np.int32))
    else:
        expert_idx = ms.Tensor(
            np.random.randint(
                0, max_expert_num, size=(
                    num_rows, k)).astype(
                        np.int32))

    # 新增入参
    scale = ms.Tensor(np.ones(2).astype(np.float32))
    offset = ms.Tensor(np.ones(2).astype(np.float32))
    quant_mode = 0

    net = MoeInitRoutingV2Net()
    result = net(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                 expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag,
                 quant_mode, scale, offset)

    print("------ result is -------", result)

def moe_init_routing_quantV2_test():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    #context.set_context(save_graphs=1, save_graphs_path="./moe_init_routing_quantV2")

    num_rows = 100
    h = 256
    k = 20
    expert_num = 20
    active_num = 0
    expert_capacity = 50
    drop_pad_mode = 0
    expert_tokens_count_or_cumsum_flag = 0
    expert_tokens_before_capacity_flag = False
    max_expert_num = 1024

    x = ms.Tensor(np.random.uniform(-1, 1, size=(num_rows, h)
                                    ).astype(np.float32), ms.float32)

    if drop_pad_mode == 1 or (
            drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag > 0):
        expert_idx = ms.Tensor(
            np.random.randint(
                0, expert_num, size=(
                    num_rows, k)).astype(
                        np.int32))
    else:
        expert_idx = ms.Tensor(
            np.random.randint(
                0, max_expert_num, size=(
                    num_rows, k)).astype(
                        np.int32))

    # 新增入参
    #scale = ms.Tensor(np.ones(1).astype(np.float32))
    #offset = ms.Tensor(np.ones(1).astype(np.float32))
    scale = ms.Tensor(np.ones(1), ms.float32)
    offset = ms.Tensor(np.ones(1), ms.float32)
    quant_mode = 0

    net = MoeInitRoutingQuantV2Net()
    result = net(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                 expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag,
                 quant_mode, scale, offset)

    print("------ result is -------", result)

def real_moe_init_routing_quantV2_test(num_rows, h, k, expert_num, active_num, drop_pad_mode,
                                       expert_tokens_count_or_cumsum_flag,
                                       expert_tokens_before_capacity_flag, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    context.set_context(
        save_graphs=1,
        save_graphs_path="./moe_init_routing_quantV2")

    max_expert_num = 1024
    x = ms.Tensor(np.random.uniform(-1, 1, size=(num_rows, h)
                                    ).astype(np.float32), ms.float32)

    if drop_pad_mode == 1 or (
            drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag > 0):
        expert_idx = ms.Tensor(
            np.random.randint(
                0, expert_num, size=(
                    num_rows, k)).astype(
                        np.int32))
    else:
        expert_idx = ms.Tensor(
            np.random.randint(
                0, max_expert_num, size=(
                    num_rows, k)).astype(
                        np.int32))

    # 新增入参
    scale = ms.Tensor(np.ones(1).astype(np.float32))
    offset = ms.Tensor(np.ones(1).astype(np.float32))
    quant_mode = 0

    net = MoeInitRoutingQuantV2Net()
    result = net(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                 expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag,
                 quant_mode, scale, offset)

    print("------ result is -------", result)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_moe_init_routing_quantv2_case0():
    """
    Feature: Test the moe_init_routing_quant_v2 in drop/pad & dynamic/static quant mode
    Description: Test the moe_init_routing_quant_v2 ops in Ascend backend
    Expectation: Run success
    """
    moe_init_routing_quantV2_test()
