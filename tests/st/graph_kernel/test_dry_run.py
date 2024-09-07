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

import os
import numpy as np
import mindspore.ops as ops
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from tests.st.graph_kernel.gk_utils import AssertGKEnable
from tests.mark_utils import arg_mark


class Net(Cell):
    def construct(self, x0, x1):
        for _ in range(150):
            x0 = ops.add(x0, x1)
        return x0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dry_run_level1():
    """
    Feature: test dvm dry run
    Description: MS_SIMULATION_LEVEL=1
    Expectation: no aclrtMallocHost error
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "1"
    os.environ["RANK_ID"] = "0"
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O1"})
    np.random.seed(1)
    x0 = np.random.normal(0, 1, (1,)).astype(np.float32)
    x1 = np.random.normal(0, 1, (1,)).astype(np.float32)
    x0_ms = Tensor(x0)
    x1_ms = Tensor(x1)
    with AssertGKEnable(True):
        net = Net()
        _ = net(x0_ms, x1_ms)
    os.environ.pop("MS_SIMULATION_LEVEL")
    os.environ.pop("RANK_SIZE")
    os.environ.pop("RANK_ID")
