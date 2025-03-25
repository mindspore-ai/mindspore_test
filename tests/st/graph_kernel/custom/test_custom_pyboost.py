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
""" tests_custom_pyboost """

import numpy as np
import mindspore as ms
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pyboost_aclnn():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("my_ops", ['jit_test_files/pyboost_aclnn_sum.cpp'], backend="Ascend").load()
    x = np.random.rand(4, 5, 6).astype(np.float32)
    expect = np.sum(np.abs(x), 1, keepdims=True)
    output = my_ops.npu_reduce_sum(ms.Tensor(x), (1,), True)
    assert np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)
