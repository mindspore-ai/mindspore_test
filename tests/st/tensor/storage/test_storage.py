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
import mindspore as ms
from mindspore import Tensor

import pytest
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_get_storage():
    """
    Feature: get_storage
    Description: Verify the result of get_storage
    Expectation: success
    """
    a = Tensor(1.0)
    b = a * 1
    storage = b.untyped_storage()

    assert storage.size() == 4
    assert storage.nbytes() == 4


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_resize():
    """
    Feature: get_storage
    Description: Verify the result of get_storage
    Expectation: success
    """
    a = Tensor(1.0)
    b = a * 1
    storage = b.untyped_storage()

    assert storage.size() == 4
    assert storage.nbytes() == 4

    storage.resize_(0)

    assert storage.size() == 0
    assert storage.nbytes() == 0
    assert storage.data_ptr() == 0

    storage.resize_(4)
    assert storage.size() == 4
    assert storage.nbytes() == 4
    assert storage.data_ptr() != 0

    storage.resize_(12)
    assert storage.size() == 12
    assert storage.nbytes() == 12
    assert storage.data_ptr() != 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_copy():
    """
    Feature: get_storage
    Description: Verify the result of get_storage
    Expectation: success
    """
    a = Tensor(2.0)
    b = a * 1
    c = a * 3
    storage_b = b.untyped_storage()
    storage_c = c.untyped_storage()

    assert b.item() == 2.0

    storage_b.copy_(storage_c)

    assert b.item() == 6.0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_no_device_exception():
    """
    Feature: get_storage
    Description: Verify the result of get_storage
    Expectation: success
    """
    a = Tensor(2.0)

    with pytest.raises(RuntimeError, match="Current Tensor has no device!"):
        a.untyped_storage()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_storage_cpu_exception():
    """
    Feature: get_storage
    Description: Verify the result of get_storage
    Expectation: success
    """
    ms.set_context(device_target="CPU")
    a = Tensor(2.0)
    a = a * 1

    with pytest.raises(RuntimeError, match="The current Storage does not yet support CPU"):
        a.untyped_storage()
