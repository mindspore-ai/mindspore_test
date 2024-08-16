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
import json
import shutil
import tempfile
from tests.mark_utils import arg_mark


def cleanup():
    kernel_meta_path = os.path.join(os.getcwd(), "kernel_data")
    cache_path = os.path.join(os.getcwd(), "__pycache__")
    if os.path.exists(kernel_meta_path):
        shutil.rmtree(kernel_meta_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)


class TestDynamicProfilerMonitor:
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    cfg_dir = tempfile.mkdtemp(prefix='dyn_prof_cfg', dir='/tmp')
    cfg_path = os.path.join(cfg_dir, 'profiler_config.json')

    @classmethod
    def setup_class(cls):
        """Run begin all test case start."""
        cleanup()

    @staticmethod
    def teardown():
        """Run after each test case end."""
        cleanup()
        if os.path.exists(TestDynamicProfilerMonitor.data_path):
            shutil.rmtree(TestDynamicProfilerMonitor.data_path)
        if os.path.exists(TestDynamicProfilerMonitor.cfg_dir):
            shutil.rmtree(TestDynamicProfilerMonitor.cfg_dir)

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_ascend_profiler(self):
        data_cfg = {
            "start_step": 10,
            "stop_step": 10,
            "aicore_metrics": -1,
            "profiler_level": 1,
            "profile_framework": -1,
            "analyse_mode": -1,
            "profile_memory": False,
            "parallel_strategy": False,
            "data_process": False,
            "data_simplification": True,
            "is_valid": False
        }

        with open(self.cfg_path, 'w') as f:
            json.dump(data_cfg, f, indent=4)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        status = os.system(
            f"""
               python ./run_net_with_dynamic_profiler.py --cfg_path={self.cfg_dir} --output_path={self.data_path}
            """
        )
        assert status == 0
        profiler_path = os.path.join(self.data_path, f"rank{rank_id}_start10_stop10")
        assert os.path.exists(profiler_path)
