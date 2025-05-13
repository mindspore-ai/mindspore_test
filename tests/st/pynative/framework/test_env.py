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

import os
import pytest
from mindspore import Tensor, ops, context
from tests.mark_utils import arg_mark

class EnvContext:
    def __init__(self, enter_callback, exit_callback):
        self.enter_callback = enter_callback
        self.exit_callback = exit_callback

    def __enter__(self):
        self.enter_callback()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_callback()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_disable_auto_h2d():
    """
    Feature: Pynative disable auto h2d copy.
    Description: Test MS_DEV_DISABLE_AUTO_H2H env.
    Expectation: run success
    """
    def set_env():
        os.environ["MS_DEV_DISABLE_AUTO_H2D"] = "1"

    def unset_env():
        os.environ.pop("MS_DEV_DISABLE_AUTO_H2D", None)

    x = Tensor(1.0)
    with EnvContext(set_env, unset_env):
        context.set_context(device_target="Ascend")
        with pytest.raises(RuntimeError):
            y = x + 1
            assert y == 2

        with pytest.raises(RuntimeError):
            y = ops.assign(x, 2)
            assert y == 2

        context.set_context(device_target="CPU")
        y = x + 1
        assert y == 2

        y = ops.assign(x, 2)
        assert y == 2
