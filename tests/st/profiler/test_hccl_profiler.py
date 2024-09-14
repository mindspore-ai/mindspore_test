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
import shutil
from mindspore import context
from tests.mark_utils import arg_mark
from file_check import FileChecker


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_hccl_allreduce():
    """
    Feature: profiler hccl operator test.
    Description: msrun hccl all_reduce 2P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = "on"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --join=True pytest -s hccl_profiler.py")
    assert return_code == 0
    rank_0_path = os.path.join(os.getcwd(), "profiler_hccl_data_0")
    rank_1_path = os.path.join(os.getcwd(), "profiler_hccl_data_1")
    FileChecker.check_timeline_values(f"{rank_0_path}/profiler/ascend_timeline_display_0.json", "name",
                                      ["hcom_allReduce*", "aclnnAdd_AddAiCore_Add"], True)
    FileChecker.check_timeline_values(f"{rank_1_path}/profiler/ascend_timeline_display_1.json", "name",
                                      ["hcom_allReduce*", "aclnnAdd_AddAiCore_Add"], True)
    op_dict = {"full_kernel_name": ["aclnnAdd_AddAiCore_Add", "hcom_allReduce*"]}
    FileChecker.check_csv_items(f"{rank_0_path}/profiler/aicore_intermediate_0_detail.csv", op_dict, fuzzy_match=True)
    FileChecker.check_csv_items(f"{rank_1_path}/profiler/aicore_intermediate_1_detail.csv", op_dict, fuzzy_match=True)
    FileChecker.check_file_line_count(f"{rank_0_path}/profiler/aicore_intermediate_0_detail.csv", 5)
    FileChecker.check_file_line_count(f"{rank_1_path}/profiler/aicore_intermediate_1_detail.csv", 5)
    if os.path.isdir(rank_0_path):
        shutil.rmtree(rank_0_path)
    if os.path.isdir(rank_0_path):
        shutil.rmtree(rank_1_path)
