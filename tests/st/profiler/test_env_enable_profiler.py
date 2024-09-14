# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark


def cleanup():
    data_path = os.path.join(os.getcwd(), "data")
    kernel_meta_path = os.path.join(os.getcwd(), "kernel_data")
    cache_path = os.path.join(os.getcwd(), "__pycache__")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    if os.path.exists(kernel_meta_path):
        shutil.rmtree(kernel_meta_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)


class CheckProfilerFiles:
    def __init__(self, device_id, rank_id, profiler_path, device_target, profile_framework=None, with_stack=False):
        """Arges init."""
        self.device_id = device_id
        self.rank_id = rank_id
        self.profiler_path = profiler_path
        self.device_target = device_target
        if device_target == "Ascend":
            self._check_d_profiling_file()
            self._check_host_profiling_file(profile_framework=profile_framework)
        elif device_target == "GPU":
            self._check_gpu_profiling_file()
            self._check_host_profiling_file(profile_framework=profile_framework)
        else:
            self._check_cpu_profiling_file()

    def _check_gpu_profiling_file(self):
        """Check gpu profiling file."""
        op_detail_file = self.profiler_path + f'gpu_op_detail_info_{self.device_id}.csv'
        op_type_file = self.profiler_path + f'gpu_op_type_info_{self.device_id}.csv'
        activity_file = self.profiler_path + f'gpu_activity_data_{self.device_id}.csv'
        timeline_file = self.profiler_path + f'gpu_timeline_display_{self.device_id}.json'
        getnext_file = self.profiler_path + f'minddata_getnext_profiling_{self.device_id}.txt'
        pipeline_file = self.profiler_path + f'minddata_pipeline_raw_{self.device_id}.csv'
        framework_file = self.profiler_path + f'gpu_framework_{self.device_id}.txt'

        gpu_profiler_files = (op_detail_file, op_type_file, activity_file,
                              timeline_file, getnext_file, pipeline_file, framework_file)
        for file in gpu_profiler_files:
            assert os.path.isfile(file)

    def _check_d_profiling_file(self):
        """Check Ascend profiling file."""
        aicore_file = self.profiler_path + f'aicore_intermediate_{self.rank_id}_detail.csv'
        # step_trace_file = self.profiler_path + f'step_trace_raw_{self.rank_id}_detail_time.csv'
        timeline_file = self.profiler_path + f'ascend_timeline_display_{self.rank_id}.json'
        aicpu_file = self.profiler_path + f'aicpu_intermediate_{self.rank_id}.csv'
        minddata_pipeline_file = self.profiler_path + f'minddata_pipeline_raw_{self.rank_id}.csv'
        queue_profiling_file = self.profiler_path + f'device_queue_profiling_{self.rank_id}.txt'
        # memory_file = self.profiler_path + f'memory_usage_{self.rank_id}.pb'

        d_profiler_files = (aicore_file, timeline_file, aicpu_file,
                            minddata_pipeline_file, queue_profiling_file)
        for file in d_profiler_files:
            assert os.path.isfile(file)

    def _check_cpu_profiling_file(self):
        """Check cpu profiling file."""
        op_detail_file = self.profiler_path + f'cpu_op_detail_info_{self.device_id}.csv'
        op_type_file = self.profiler_path + f'cpu_op_type_info_{self.device_id}.csv'
        timeline_file = self.profiler_path + f'cpu_op_execute_timestamp_{self.device_id}.txt'

        cpu_profiler_files = (op_detail_file, op_type_file, timeline_file)
        for file in cpu_profiler_files:
            assert os.path.isfile(file)

    def _check_host_profiling_file(self, profile_framework='all'):
        dataset_csv = os.path.join(self.profiler_path, f'dataset_{self.rank_id}.csv')
        if profile_framework in ['all', 'time']:
            assert os.path.isfile(dataset_csv)
        else:
            assert not os.path.exists(dataset_csv)


class TestEnvEnableProfiler:
    profiler_path = os.path.join(os.getcwd(), f'data/profiler/')
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0

    @classmethod
    def setup_class(cls):
        """Run begin all test case start."""
        cleanup()

    @staticmethod
    def teardown():
        """Run after each test case end."""
        cleanup()

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
    def test_ascend_profiler(self):
        status = os.system(
            """export MS_PROFILER_OPTIONS='{"start":true, "profile_memory":true, "profile_framework":"all", "data_process":true, "with_stack":false}';
               python ./run_net.py --target=Ascend --mode=0;
            """
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "Ascend", "all")
        assert status == 0

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
    def test_host_profiler_none(self):
        status = os.system(
            """export MS_PROFILER_OPTIONS='{"start":true, "profile_memory":true, "data_process":true}';
               python ./run_net.py --target=Ascend --mode=0;
            """
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "Ascend", None, False)
        assert status == 0

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
    def test_host_profiler_time(self):
        status = os.system(
            """export MS_PROFILER_OPTIONS='{"start":true, "profile_memory":true, "profile_framework":"time", "data_process":true}';
               python ./run_net.py --target=Ascend --mode=0;
            """
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "Ascend", "time")
        assert status == 0
