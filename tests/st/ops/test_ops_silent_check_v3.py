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
import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
from mindspore.device_context.cpu.op_tuning import threads_num


def silent_check_v3(val, max_val, avg, input_grad, step,
                    c_thresh_l1=1000000., c_thresh_l2=10000., beta1=0., npu_asd_detect=1):
    op = ops.auto_generate.silent_check_v3_op
    _, new_input_grad, _, result = op(val, max_val, avg, input_grad, step,
                                      c_thresh_l1, c_thresh_l2, beta1, npu_asd_detect)
    return avg, new_input_grad, step, result


@test_utils.run_with_cell
def silent_check_v3_forward_func(val, max_val, avg, input_grad, step,
                                 c_thresh_l1=1000000., c_thresh_l2=10000., beta1=0., npu_asd_detect=1):
    return silent_check_v3(val, max_val, avg, input_grad, step,
                           c_thresh_l1, c_thresh_l2, beta1, npu_asd_detect)


def set_mode(mode):
    """
    set_mode
    """
    if mode == "kbk":
        context.set_context(mode=context.GRAPH_MODE, jit_config={'jit_level': 'O0'})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


def generate_inputs():
    """
    generate_inputs
    """
    val = Tensor(np.random.rand(1), ms.float32)
    max_val = Tensor(np.random.rand(1), ms.float32)
    avg = Tensor(np.random.rand(1), ms.float32)
    input_grad = Tensor(np.random.rand(2, 5).astype(np.float32))
    step = Tensor(np.random.randint(1, 10, size=[1]), ms.int64)
    c_thresh_l1 = 1000000.
    c_thresh_l2 = 10000.
    beta1 = 0.
    npu_asd_detect = 1
    return [val, max_val, avg, input_grad, step,
            c_thresh_l1, c_thresh_l2, beta1, npu_asd_detect]


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", ["kbk", "pyboost"])
def test_silent_check_v3_static_shape(mode):
    """
    Feature: SilentCheckV3.
    Description: test op SilentCheckV3.
    Expectation: expect correct result.
    """
    set_mode(mode)
    inputs = generate_inputs()
    avg, input_grad, step = inputs[2:5]
    print(f"before avg:\n{avg}\ninput_grad:\n{input_grad}\nstep:\n{step}.")
    outs = silent_check_v3_forward_func(*inputs)
    print(f"after silent check, avg:\n{outs[0]}\ninput_grad:\n{outs[1]}"
          f"\nstep:\n{outs[2]}\nresult:\n{outs[3]}.")


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
def test_silent_check_v3_dyn_shape():
    """
    Feature: SilentCheckV3.
    Description: test op SilentCheckV3.
    Expectation: expect correct result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    TEST_OP(
        silent_check_v3_forward_func,
        [
            generate_inputs(),
            generate_inputs(),
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True,
                     'disable_grad': True,
                     'deterministic_use_origin_inputs': True},
        inplace_update=True
    )
