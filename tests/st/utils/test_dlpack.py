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
from tests.mark_utils import arg_mark
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.utils.dlpack import from_dlpack, to_dlpack



@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dlpack_npu_tensor_conversion():
    """
    Feature: test dlpack for npu tensor
    Description: test from_dlpack and to_dlpack for npu tensor
    Expectation: success
    """
    ms.set_context(device_target="Ascend")
    x = Tensor(np.array([1, 2, 3]), ms.float32)
    x = x.add_(1)
    x_ptr = x.data_ptr()
    dlpack_x = to_dlpack(x)
    y = from_dlpack(dlpack_x)
    y_ptr = y.data_ptr()
    assert x_ptr == y_ptr
    assert np.allclose(x.asnumpy(), y.asnumpy())
