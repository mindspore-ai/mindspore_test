# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import numpy as np

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [0, 1])
def test_bitwise_and(mode):
    """
    Feature: Tensor.bitwise_and
    Description: Verify the result of bitwise_and
    Expectation: success
    """
    os.environ["MS_TENSOR_METHOD_BOOST"] = "1"
    import mindspore as ms
    import mindspore.nn as nn

    class Net(nn.Cell):
        def construct(self, x, y):
            output = x.bitwise_and(y)
            return output

    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32))
    y = ms.Tensor(np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int32))
    expect_output = np.array([[1, 0, 3], [4, 5, 4], [3, 0, 1]], dtype=np.int32)
    out = net(x, y)
    assert np.allclose(out.asnumpy(), expect_output)
    out2 = x & y
    assert np.allclose(out2.asnumpy(), expect_output)
    del os.environ["MS_TENSOR_METHOD_BOOST"]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [0, 1])
def test_bitwise_and_with_bool(mode):
    """
    Feature: Tensor.bitwise_and
    Description: Verify the result of bitwise_and with bool type
    Expectation: success
    """
    os.environ["MS_TENSOR_METHOD_BOOST"] = "1"
    import mindspore as ms
    import mindspore.nn as nn

    class Net(nn.Cell):
        def construct(self, x, y):
            output = x.bitwise_and(y)
            return output

    ms.set_context(mode=mode, jit_config={"jit_level": "O1"})
    net = Net()
    x = ms.Tensor(np.array([[True, False, True], [False, True, False], [True, False, True]], dtype=np.bool_))
    y = ms.Tensor(np.array([[True, True, False], [False, True, True], [True, True, False]], dtype=np.bool_))
    expect_output = np.array([[True, False, False], [False, True, False], [True, False, False]], dtype=np.bool_)
    out = net(x, y)
    assert np.allclose(out.asnumpy(), expect_output)
    out2 = x & y
    assert np.allclose(out2.asnumpy(), expect_output)
    del os.environ["MS_TENSOR_METHOD_BOOST"]
