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
"""Test custom aicpu profiling."""
import tempfile
import numpy as np

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Profiler
from mindspore import Tensor
from tests.mark_utils import arg_mark
from model_zoo import CustomAICpuNet
from file_check import FileChecker


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_collect_custom_aicpu():
    """
    Feature: Profiling can collect custom aicpu operators
    Description: Test profiling can collect custom aicpu operators on ascend
    Expectation: The file aicpu_intermediate_*.csv generated successfully and s1 == s2
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory(suffix="profiler_ai_cpu") as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        net = CustomAICpuNet()
        net(Tensor(np.random.random((6,)), mstype.float64))
        profiler.analyse()
        op_dict = {"kernel_type": ["Cast", "Select", "Xlogy"]}
        FileChecker.check_csv_items(f"{tmpdir}/profiler/aicpu_intermediate_0.csv", op_dict, fuzzy_match=False)
