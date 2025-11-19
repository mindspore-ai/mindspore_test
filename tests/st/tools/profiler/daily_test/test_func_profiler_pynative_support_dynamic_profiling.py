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
"""dynamic profiler test."""
import os
import tempfile
import json

from mindspore.mint.distributed import get_rank
from tests.mark_utils import arg_mark
from tests.st.tools.profiler.daily_test.profiler_check import MSProfilerChecker


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_func_profiler_dynamic_profiler_monitor_start_1_19_change_1_19():
    """
    Feature: Profiler Dynamic Steps
    Description: Test PyNative dynamic profiling for continuous 1–19 and 10_15.
    Expectation: Generate profiling data and validate expected step IDs.
    """
    data_cfg = {
        "start_step": 6,
        "stop_step": 7,
        "aic_metrics": "ArithmeticUtilization",
        "profiler_level": "LevelNone",
        "activities": ["CPU", "NPU"],
        "analyse": True,
        "record_shapes": True,
        "analyse_mode": 0,
        "with_stack": True,
        "host_sys": ["cpu"],
        "mstx_domain_include": [],
        "mstx_domain_exclude": [],
        "sys_interconnection": True,
        "sys_io": True,
        "export_type": ["text", "db"],
        "parallel_strategy": True,
        "data_simplification": True,
    }

    with tempfile.TemporaryDirectory(suffix="1_19") as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        with open(cfg_path, 'w', encoding='utf-8') as file:
            json.dump(data_cfg, file, ensure_ascii=False, indent=4)

        rank_id = get_rank()
        dir_list = [f"rank{rank_id}_start6_stop7"]
        ret = os.system(f"python run_net_with_dynamic_profiler.py --output_path {tmpdir} --cfg_path {tmpdir}")
        assert ret == 0
        for i in range(1):
            prof_config = {"output_path": os.path.join(tmpdir, dir_list[i]),
                           "profile_memory": False,
                           "profile_level": 0,
                           "record_shapes": True,
                           "sys_io": True,
                           "sys_interconnection": True,
                           "pynative_step": True,
                           "export_type": "dbtext"}
            profiler_check = MSProfilerChecker(prof_config, 1, check_step_id=[1, 2])
            profiler_check()
