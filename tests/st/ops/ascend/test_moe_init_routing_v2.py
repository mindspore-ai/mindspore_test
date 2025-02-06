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
import pytest
import mindspore as ms
from mindspore import context
from mindspore import jit, JitConfig
from mindspore.ops.auto_generate import MoeInitRoutingV2
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

moe_init_routing_v2_op = MoeInitRoutingV2()

def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

def moe_init_routing_v2_exec(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                             expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
    num_rows = x.shape[0]
    hidden_size = x.shape[-1]
    k = expert_idx.shape[-1]
    sorted_row_idx = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind="stable")
    sorted_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1)
    if drop_pad_mode == 1 and expert_num <= 0:
        print("expert num can not be 0")
        return
    
    expert_tokens_count_or_cumsum = None
    expert_tokens_before_capacity = None
    # expert_token_idx
    expert_idx_hist, _ = np.histogram(sorted_expert_idx, bins=expert_num, range=(0, expert_num - 1))
    expert_token_idx = np.cumsum(expert_idx_hist)
    if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
        expert_tokens_before_capacity = expert_idx_hist.astype("int32")
    if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
        expert_tokens_count_or_cumsum = expert_token_idx.astype("int32")
    elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
        expert_tokens_count_or_cumsum = expert_idx_hist.astype("int32")
    
    if drop_pad_mode == 0:
        expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
        expanded_row_idx[sorted_row_idx] = np.arange(sorted_row_idx.shape[-1], dtype=np.int32)

        if active_num == 0:
            active_num = num_rows * k
        else:
            active_num = min(active_num, num_rows * k)
        expanded_x = x[sorted_row_idx[:active_num] // k, :]
    else:
        adapter_capacity(sorted_row_idx, sorted_expert_idx, expert_capacity)
        sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
        offset = 0
        last_expert_id = 0
        for i, val in enumerate(sorted_row_idx):
            if val != -1:
                if last_expert_id != sorted_expert_idx[i]:
                    offset = 0
                    last_expert_id = sorted_expert_idx[i]
                sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = sorted_row_idx[i]
                offset = offset + 1
        
        # expanded_row_idx
        expanded_row_idx = np.full(sorted_row_idx.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i

        # expanded_x
        expanded_x = np.full((expert_num * expert_capacity, hidden_size), 0, dtype=x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = x[val // k]
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)

    return expanded_x, expanded_row_idx.astype("int32"), expert_tokens_count_or_cumsum, expert_tokens_before_capacity

@test_utils.run_with_cell
def moe_init_routing_v2_forward_func(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                                     expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
    return moe_init_routing_v2_op(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                                     expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_moe_init_routing_v2_case0(mode):
    """
    Feature: Test the moe_init_routing_v2 forward in drop/pad mode
    Description: Test the moe_init_routing_v2 ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    num_rows = 10
    h = 200
    k = 2
    active_num = num_rows
    expert_num = 10
    drop_pad_mode = 1
    expert_tokens_count_or_cumsum_flag = 0
    expert_tokens_before_capacity_flag = False
    expert_capacity = 6

    # numpy input
    x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(np.float16)
    expert_idx = np.random.randint(0, 10, size=(num_rows, k)).astype(np.int32)

    expanded_x, expanded_row_idx, _, _ = moe_init_routing_v2_exec(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                             expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag)

    # tensor input
    x_tensor = ms.Tensor(x, ms.float16)
    expert_idx_tensor = ms.Tensor(expert_idx, ms.int32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        expanded_x_ms, expanded_row_idx_ms, _, _ = \
            moe_init_routing_v2_forward_func(x_tensor, expert_idx_tensor, active_num, expert_capacity, expert_num, drop_pad_mode,
                                     expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        expanded_x_ms, expanded_row_idx_ms, _, _ = \
            (jit(moe_init_routing_v2_forward_func, jit_config=JitConfig(jit_level="O0")))(x_tensor, expert_idx_tensor, active_num, expert_capacity, expert_num, drop_pad_mode,
                                     expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag)

    np.testing.assert_allclose(expanded_x_ms.asnumpy(), expanded_x, rtol=1e-3)
    np.testing.assert_allclose(expanded_row_idx_ms.asnumpy(), expanded_row_idx, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_moe_init_routing_v2_dynamic():
    """
    Feature: test moe_init_routing_v2
    Description: dynamic shape and rank
    Expectation: success
    """
    num_rows = 10
    h = 200
    k = 2
    active_num = num_rows
    expert_num = 10
    drop_pad_mode = 1
    expert_tokens_count_or_cumsum_flag = 0
    expert_tokens_before_capacity_flag = True
    expert_capacity = 6

    # tensor input
    x_1 = ms.Tensor(np.random.uniform(-1, 1, size=(num_rows, h)).astype(np.float16), ms.float16)
    expert_idx_1 = ms.Tensor(np.random.randint(0, 10, size=(num_rows, k)).astype(np.int32), ms.int32)

    x_2 = ms.Tensor(np.random.uniform(-1, 1, size=(num_rows, h)).astype(np.float16), ms.float16)
    expert_idx_2 = ms.Tensor(np.random.randint(0, 10, size=(num_rows, k)).astype(np.int32), ms.int32)

    TEST_OP(moe_init_routing_v2_forward_func,
            [[x_1, expert_idx_1, active_num, expert_capacity, expert_num, drop_pad_mode, 
              expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag], [x_2, expert_idx_2, active_num,
              expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag]],
            'moe_init_routing_v2', disable_mode=["GRAPH_MODE"], disable_grad=True, disable_input_check=True,
            ignore_output_index=2, disable_nontensor_dynamic_type='BOTH')
