# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
"""test cpu profiler"""
import os
import shutil
import sys
import tempfile
import numpy as np

import mindspore.context as context
from mindspore import Tensor
from mindspore import Profiler
from tests.mark_utils import arg_mark
from model_zoo import TinyAddNet
from file_check import FileChecker


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_cpu_profiling():
    """
    Feature: Profiling can collect custom aicpu operators
    Description: Test profiling can collect custom aicpu operators on ascend
    Expectation: The file aicpu_intermediate_*.csv generated successfully and s1 == s2
    """
    if sys.platform != 'linux':
        return
    data_path = os.path.join(os.getcwd(), 'data_cpu')
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    with tempfile.TemporaryDirectory(suffix="profiler_cpu") as tmpdir:
        profiler = Profiler(output_path=tmpdir, data_simplification=False)
        x = np.random.randn(1, 3, 3, 4).astype(np.float32)
        y = np.random.randn(1, 3, 3, 4).astype(np.float32)
        add = TinyAddNet()
        add(Tensor(x), Tensor(y))
        profiler.analyse()

        op_detail_file = f"{tmpdir}/profiler/cpu_op_detail_info_{rank_id}.csv"
        op_type_file = f"{tmpdir}/profiler/cpu_op_type_info_{rank_id}.csv"
        timeline_file = f"{tmpdir}/profiler/cpu_op_execute_timestamp_{rank_id}.txt"

        op_dict = {"full_op_name": "Default/Add-op*"}
        FileChecker.check_csv_items(op_detail_file, op_dict, fuzzy_match=True)
        op_dict = {"op_type": "Add"}
        FileChecker.check_csv_items(op_type_file, op_dict, fuzzy_match=False)
        FileChecker.check_txt_not_empty(timeline_file)
        profiler.op_analyse(op_name="Add")
