# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""CPU platform profiler."""
from mindspore import log as logger
from mindspore.profiler.common.registry import PROFILERS
from mindspore.profiler.common.constant import DeviceTarget
from mindspore.profiler.platform_profiler.profiler_interface import ProfilerInterface


@PROFILERS.register_module(DeviceTarget.CPU.value)
class CpuProfiler(ProfilerInterface):
    """
    CPU platform profiler
    """

    def __init__(
            self,
            op_time: bool = True,
            with_stack: bool = False,
            data_process: bool = False,
            output_path: str = "./data",
            profile_memory: bool = False,
            profile_framework: str = "all",
            **kwargs
    ) -> None:
        super().__init__()

    def start(self) -> None:
        """Start profiling."""
        logger.info("CpuProfiler start.")

    def stop(self) -> None:
        """Stop profiling."""
        logger.info("CpuProfiler stop.")

    def analyse(self, pretty=False, step_list=None, mode="sync", rank_id=None) -> None:
        """Analyse profiling data."""
        logger.info("CpuProfiler analyse.")
