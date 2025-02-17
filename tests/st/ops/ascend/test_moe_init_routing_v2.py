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
from mindspore.nn import Cell
from mindspore.ops.auto_generate import MoeInitRoutingV2
from tests.mark_utils import arg_mark


SUPPORTED_DTYPE = [ms.float16, ms.bfloat16, ms.float32]
MAX_EXPERT_NUM = 1024

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
        raise Exception("[Error] expert_num must be greater than 0 when drop pad mode is enabled")

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

class MoeInitRoutingV2Net(Cell):
    def __init__(self):
        super().__init__()
        self.net = MoeInitRoutingV2()

    def construct(self, *args):
        return self.net(*args)

class TestMoeInitRoutingV2:
    def __init__(self, test_inputs: dict):
        self.inputs_dct = test_inputs
        self.inputs = test_inputs.get("inputs", {})
        self.net = MoeInitRoutingV2Net()
        self.case_name = test_inputs.get("case_name", "None")
        self.mode = test_inputs.get("mode", ms.PYNATIVE_MODE)
        self.dtype = test_inputs.get("dtype", ms.float16)

        self.init_ctx()
        self.set_ms_inputs()
        self.np_out = {}
        self.ms_out = {}
        self.cal_np_out()
        self.cal_ms_out()
        self.compare()

    def cal_np_out(self):
        print(f"start test {self.case_name}")
        np.random.seed(0)
        num_rows = self.inputs_dct["NUM_ROWS"]
        h = self.inputs_dct["H"]
        k = self.inputs_dct["K"]
        self.ms_inputs["x"] = np.random.uniform(-1, 1, size=(num_rows, h)).astype(np.float16)
        if self.ms_inputs["drop_pad_mode"] == 1 or (self.ms_inputs["drop_pad_mode"] == 0 and \
                                                    self.ms_inputs["expert_tokens_count_or_cumsum_flag"] > 0):
            self.ms_inputs["expert_idx"] = np.random.randint(0, self.ms_inputs["expert_num"], \
                                                             size=(num_rows, k)).astype(np.int32)
        else:
            self.ms_inputs["expert_idx"] = np.random.randint(0, MAX_EXPERT_NUM, size=(num_rows, k)).astype(np.int32)
        self.np_out["expanded_x"], self.np_out["expanded_row_idx"],\
        self.np_out["expert_tokens_count_or_cumsum"],\
        self.np_out["expert_tokens_before_capacity"] = moe_init_routing_v2_exec(*tuple(self.ms_inputs.values()))

    def cal_ms_out(self):
        self.ms_inputs["x"] = ms.Tensor(self.ms_inputs["x"], self.dtype)
        self.ms_inputs["expert_idx"] = ms.Tensor(self.ms_inputs["expert_idx"], ms.int32)
        self.ms_out["expanded_x"], self.ms_out["expanded_row_idx"],\
        self.ms_out["expert_tokens_count_or_cumsum"],\
        self.ms_out["expert_tokens_before_capacity"] = self.net(*tuple(self.ms_inputs.values()))

    def compare(self):
        for expect_out, out in zip(self.np_out.values(), self.ms_out.values()):
            if expect_out is None:
                continue
            np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=5e-3)
        print(f"test success for {self.case_name}")

    def set_ms_inputs(self):
        self.ms_inputs = {
            "x": None, # tensor
            "expert_idx": None, # tensor
            "active_num": None, # int
            "expert_capacity": None, # int
            "expert_num": None, # int
            "drop_pad_mode": None, # int
            "expert_tokens_count_or_cumsum_flag": None, # int
            "expert_tokens_before_capacity_flag": None, # bool
        }
        self.ms_inputs["active_num"] = self.inputs["active_num"]
        self.ms_inputs["expert_capacity"] = self.inputs["expert_capacity"]
        self.ms_inputs["expert_num"] = self.inputs["expert_num"]
        self.ms_inputs["drop_pad_mode"] = self.inputs["drop_pad_mode"]
        self.ms_inputs["expert_tokens_count_or_cumsum_flag"] = self.inputs["expert_tokens_count_or_cumsum_flag"]
        self.ms_inputs["expert_tokens_before_capacity_flag"] = self.inputs["expert_tokens_before_capacity_flag"]
        if self.inputs_dct["is_dynamic"]:
            x_shape = [None, None]
            expert_idx_shape = [None, None]
            self.ms_inputs["x"] = ms.Tensor(shape=x_shape, dtype=self.dtype)
            self.ms_inputs["expert_idx"] = ms.Tensor(shape=expert_idx_shape, dtype=ms.int32)

            self.net.set_inputs(*tuple(self.ms_inputs.values()))
            self.ms_inputs["x"] = None
            self.ms_inputs["expert_idx"] = None

    def init_ctx(self):
        if self.dtype not in SUPPORTED_DTYPE:
            raise Exception("[Error] unsupported input dtype.")
        if self.mode == ms.PYNATIVE_MODE:
            context.set_context(mode=ms.PYNATIVE_MODE)
        elif self.mode == ms.GRAPH_MODE:
            context.set_context(mode=ms.GRAPH_MODE)
            context.set_context(jit_config={"jit_level": "O0"})
        else:
            raise Exception("[Error] unsupported mode.")


description = {
    "case_name": "Test case name",
    "mode": "Run mode, support pynative and kbk",
    "dtype": "Dtype of input x",
    "is_dynamic": "Whether enable dynamic shape",
    "NUM_ROWS": "The first dim of input x. int",
    "H": "The second dim of input x. int",
    "K": "The second dim of input expert_idx. int",
    "inputs": "Inputs of Op MoeInitRoutingV2",
    "x": "Input x, shape is [NUM_ROWS, H], dtype support float16/bfloat16/float32",
    "expert_idx": "Input expert index, shape is [NUM_ROWS, K]. The first dim must be the same as x's, dtype \
        support int32.",
    "active_num": "Number of experts. int",
    "expert_capacity": "Capacity of experts. int",
    "drop_pad_mode": "Whether enable drop/pad mode. int, 1 enable, 0 disable",
    "expert_tokens_count_or_cumsum_flag": "Control whether MoeInitRoutingV2 output expert_tokens_count_or_cumsum. int",
    "expert_tokens_before_capacity_flag": "Control whether MoeInitRoutingV2 output expert_tokens_before_capacity. bool"
}


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16, ms.float32])
def test_moe_init_routing_v2_case0(mode, dtype):
    """
    Feature: Test the moe_init_routing_v2 forward in drop/pad mode
    Description: Test the moe_init_routing_v2 ops in Ascend backend
    Expectation: Run success
    """

    test_inputs = {
        "case_name": "MoeInitRoutingV2 forward in drop/pad mode",
        "mode": mode,
        "dtype": dtype,
        "is_dynamic": False,
        "NUM_ROWS": 100,
        "H": 256,
        "K": 20,
        "inputs": {
            "x": None,
            "expert_idx": None,
            "active_num": 0,
            "expert_capacity": 50,
            "expert_num": 20,
            "drop_pad_mode": 1,
            "expert_tokens_count_or_cumsum_flag": 0,
            "expert_tokens_before_capacity_flag": False
        }
    }

    TestMoeInitRoutingV2(test_inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16, ms.float32])
def test_moe_init_routing_v2_case1(mode, dtype):
    """
    Feature: Test the moe_init_routing_v2 forward in drop/pad mode with expert_tokens_before_capacity_flag = True
    Description: Test the moe_init_routing_v2 ops in Ascend backend
    Expectation: Run success
    """

    test_inputs = {
        "case_name": "MoeInitRoutingV2 forward in drop/pad mode \
            with expert_tokens_before_capacity_flag = True",
        "mode": mode,
        "dtype": dtype,
        "is_dynamic": False,
        "NUM_ROWS": 100,
        "H": 256,
        "K": 20,
        "inputs": {
            "x": None,
            "expert_idx": None,
            "active_num": 0,
            "expert_capacity": 50,
            "expert_num": 20,
            "drop_pad_mode": 1,
            "expert_tokens_count_or_cumsum_flag": 0,
            "expert_tokens_before_capacity_flag": True
        }
    }

    TestMoeInitRoutingV2(test_inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16, ms.float32])
def test_moe_init_routing_v2_case2(mode, dtype):
    """
    Feature: Test the moe_init_routing_v2 forward in dropless mode
    Description: Test the moe_init_routing_v2 ops in Ascend backend
    Expectation: Run success
    """

    test_inputs = {
        "case_name": "MoeInitRoutingV2 forward in dropless mode",
        "mode": mode,
        "dtype": dtype,
        "is_dynamic": False,
        "NUM_ROWS": 100,
        "H": 256,
        "K": 20,
        "inputs": {
            "x": None,
            "expert_idx": None,
            "active_num": 0,
            "expert_capacity": 50,
            "expert_num": 20,
            "drop_pad_mode": 0,
            "expert_tokens_count_or_cumsum_flag": 0,
            "expert_tokens_before_capacity_flag": False
        }
    }

    TestMoeInitRoutingV2(test_inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16, ms.float32])
def test_moe_init_routing_v2_case3(mode, dtype):
    """
    Feature: Test the moe_init_routing_v2 forward in dropless mode with expert_tokens_count_or_cumsum_flag > 0
    Description: Test the moe_init_routing_v2 ops in Ascend backend
    Expectation: Run success
    """

    test_inputs = {
        "case_name": "MoeInitRoutingV2 forward in dropless mode \
            with expert_tokens_count_or_cumsum_flag > 0",
        "mode": mode,
        "dtype": dtype,
        "is_dynamic": False,
        "NUM_ROWS": 100,
        "H": 256,
        "K": 20,
        "inputs": {
            "x": None,
            "expert_idx": None,
            "active_num": 0,
            "expert_capacity": 50,
            "expert_num": 20,
            "drop_pad_mode": 0,
            "expert_tokens_count_or_cumsum_flag": 1,
            "expert_tokens_before_capacity_flag": False
        }
    }

    TestMoeInitRoutingV2(test_inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16, ms.float32])
def test_moe_init_routing_v2_case4(mode, dtype):
    """
    Feature: Test the moe_init_routing_v2 forward in drop/pad mode with dynamic shape
    Description: Test the moe_init_routing_v2 ops in Ascend backend
    Expectation: Run success
    """

    test_inputs = {
        "case_name": "MoeInitRoutingV2 forward in drop/pad mode with dynamic shape",
        "mode": mode,
        "dtype": dtype,
        "is_dynamic": True,
        "NUM_ROWS": 100,
        "H": 256,
        "K": 20,
        "inputs": {
            "x": None,
            "expert_idx": None,
            "active_num": 0,
            "expert_capacity": 50,
            "expert_num": 20,
            "drop_pad_mode": 1,
            "expert_tokens_count_or_cumsum_flag": 0,
            "expert_tokens_before_capacity_flag": False
        }
    }

    TestMoeInitRoutingV2(test_inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16, ms.float32])
def test_moe_init_routing_v2_case5(mode, dtype):
    """
    Feature: Test the moe_init_routing_v2 forward in active mode
    Description: Test the moe_init_routing_v2 ops in Ascend backend
    Expectation: Run success
    """

    test_inputs = {
        "case_name": "MoeInitRoutingV2 forward in active mode",
        "mode": mode,
        "dtype": dtype,
        "is_dynamic": False,
        "NUM_ROWS": 100,
        "H": 256,
        "K": 20,
        "inputs": {
            "x": None,
            "expert_idx": None,
            "active_num": 10,
            "expert_capacity": 50,
            "expert_num": 20,
            "drop_pad_mode": 0,
            "expert_tokens_count_or_cumsum_flag": 0,
            "expert_tokens_before_capacity_flag": False
        }
    }

    TestMoeInitRoutingV2(test_inputs)
