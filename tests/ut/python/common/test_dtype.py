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
"""Test dtype."""
import mindspore as ms

def test_dtype_equal():
    """
    Feature: Tensor dtype share one instance.
    Description: Test Tensor dtype share one instance.
    Expectation: Success.
    """
    a = ms.Tensor(1)
    assert a.dtype is ms.int64
    b = ms.Tensor(1, dtype=ms.float32)
    assert b.dtype is ms.float32
