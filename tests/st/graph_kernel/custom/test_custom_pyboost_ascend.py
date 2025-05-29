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
""" tests_custom_pyboost_ascend """

import numpy as np
import mindspore as ms
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_pyboost_atb_swiglu():
    """
    Feature: CustomOpBuilder.
    Description: Custom atb op.
    Expectation: success.
    """
    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("atb_swiglu", "jit_test_files/atb_swiglu.cpp", enable_atb=True).load()
    # the second dim of x should be >= 32
    x = np.array([[0.561, 0.684, 0.329, 0.8447, 0.2815, 0.0716, 0.3472, 0.04404,
                   0.9565, 0.9033, 0.3567, 0.33, 0.2467, 0.2993, 0.0109, 0.9243,
                   0.2163, 0.4355, 0.4707, 0.9463, 0.5156, 0.978, 0.815, 0.247,
                   0.7153, 0.677, 0.9263, 0.665, 0.353, 0.0239, 0.4363, 0.9097],
                  [0.9585, 0.1242, 0.05566, 0.642, 0.5103, 0.658, 0.704, 0.4739,
                   0.299, 0.1958, 0.2349, 0.10657, 0.2134, 0.1458, 0.4458, 0.2399,
                   0.6626, 0.4255, 0.5674, 0.5454, 0.3523, 0.5435, 0.03458, 0.912,
                   0.3064, 0.9287, 0.8633, 0.2822, 0.652, 0.1549, 0.6426, 0.004536]], dtype=np.float16)
    expect = np.array([[0.0773, 0.198, 0.0901, 0.559, 0.0827, 0.03625, 0.1658, 0.005558,
                        0.4944, 0.435, 0.1943, 0.1277, 0.0489, 0.00411, 0.002392, 0.602],
                       [0.459, 0.02806, 0.01624, 0.2295, 0.1123, 0.2357, 0.0163, 0.2664,
                        0.0526, 0.0998, 0.1132, 0.01584, 0.07697, 0.01211, 0.1747, 0.000609]], dtype=np.float16)
    output = my_ops.npu_swiglu(ms.Tensor(x), -1)
    np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_pyboost_aclnn():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("aclnn_op", ['jit_test_files/pyboost_aclnn_sum.cpp'], backend="Ascend").load()
    x = np.random.rand(4, 5, 6).astype(np.float32)
    expect = np.sum(np.abs(x), 1, keepdims=True)
    output = my_ops.npu_abs_reduce_sum(ms.Tensor(x), (1,), True)
    assert np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)
