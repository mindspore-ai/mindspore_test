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
import pytest
from mindspore import Tensor
import mindspore.common.dtype as mstype

from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_set_getstate_bypickle():
    """
    Feature: TensorPy.set_device_address
    Description: Verify the result of TensorPy.set_device_address
    Expectation: success
    """
    with pytest.raises(RuntimeError):
        seed_max = 0xffff_ffff_ffff_ffff + 1
        a = Tensor(seed_max, mstype.int64)  # pylint: disable=W0612
