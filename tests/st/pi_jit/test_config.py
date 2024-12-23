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
"""test jit config"""
from mindspore import Tensor, jit
from .share.utils import assert_executed_by_graph_mode
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_config_disable_pijit():
    """
    Feature: Jit config
    Description: Jit config
    Expectation: The result match
    """
    @jit(mode="PIJit", jit_config={'_disable_pijit':lambda args, kwds: args[0] > 1})
    def func(x, y):
        return x + y

    for i in range(10):
        a = Tensor([i])
        func(a, a)

    assert_executed_by_graph_mode(func, call_count=2)
