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
""" test Activations internal op """
import os
import numpy as np
import pytest
from mindspore.ops import silu
from mindspore.common.np_dtype import bfloat16
from mindspore import Tensor, context
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from st_utils import custom_compare


@test_utils.run_with_cell
def silu_forward_func(x):
    return silu(x)

def np_silu(x):
    return x * (1 / (1 + np.exp(-x)))


def _test_silu(shape, np_dtype, is_dynamic=False, mode=context.GRAPH_MODE):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    context.set_context(mode=mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    input_np = np.random.randn(*shape).astype(np_dtype)
    expected = np_silu(input_np)

    input_data = Tensor(input_np)
    if is_dynamic:
        input_dyn = Tensor(shape=[None] * len(shape), dtype=input_data.dtype)
        test_cell = test_utils.to_cell_obj(silu_forward_func)
        test_cell.set_inputs(input_dyn)
        output = test_cell(input_data)
    else:
        output = silu_forward_func(input_data)
    output_np = output.asnumpy()
    res = custom_compare(output_np, expected, input_data.dtype)
    assert res, f"SiLU compare failed"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('shape', [(1, 2048), (16, 128, 32)])
@pytest.mark.parametrize('dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('is_dynamic', [True, False])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_swiglu(shape, dtype, is_dynamic, mode):
    """
    Feature: Test SiLU.
    Description: Test SiLU internal op.
    Expectation: Success.
    """
    _test_silu(shape, dtype, is_dynamic, mode)
