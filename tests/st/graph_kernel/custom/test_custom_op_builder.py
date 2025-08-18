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
""" tests_custom_op_builder """

from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark
import pytest


@pytest.mark.parametrize("kwargs,expect_type,expect_msg", [
    ({"name": 123, "sources": "op.cc"}, TypeError, "name"),
    ({"name": "Op", "sources": 123}, TypeError, "sources"),
    ({"name": "Op", "sources": ["a.cc", 123]}, TypeError, "sources"),
    ({"name": "Op", "sources": "a.cc", "backend": "GPU"}, ValueError, "backend"),
    ({"name": "Op", "sources": "a.cc", "include_paths": ["inc", 123]}, TypeError, "include_paths"),
    ({"name": "Op", "sources": "a.cc", "cflags": 123}, TypeError, "cflags"),
    ({"name": "Op", "sources": "a.cc", "ldflags": 123}, TypeError, "ldflags"),
    ({"name": "Op", "sources": "a.cc", "debug_mode": "true"}, TypeError, "debug_mode"),
    ({"name": "Op", "sources": "a.cc", "op_def": 123}, TypeError, "op_def"),
    ({"name": "Op", "sources": "a.cc", "op_def": ["x.yaml", 123]}, TypeError, "op_def"),
    ({"name": "Op", "sources": "a.cc", "op_doc": 123}, TypeError, "op_doc"),
    ({"name": "Op", "sources": "a.cc", "backend": "CPU", "enable_atb": True}, ValueError, "enable_atb"),
])
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_custom_op_builder_invalid_args(kwargs, expect_type, expect_msg):
    """
    Feature: test custom op parameter validation
    Description: pass illegal arguments and expect exceptions
    Expectation: raises the specified exception type with matching message
    """
    with pytest.raises(expect_type, match=f".*{expect_msg}.*"):
        CustomOpBuilder(**kwargs)
