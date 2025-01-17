# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
@File   : conftest.py
@Desc   : common fixtures for pytest
"""

import pytest
import sys

from mindspore._c_expression import update_pijit_default_config


def pytest_runtest_setup(item):
    if sys.version_info >= (3, 11):
        pytest.skip(reason="Skipping PIJit tests for Python >= 3.11.")
    update_pijit_default_config(compile_with_try=False)
