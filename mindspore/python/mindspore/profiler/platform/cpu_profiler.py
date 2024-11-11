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
from mindspore.profiler.common.constant import DeviceTarget, ProfilerActivity
import mindspore._c_expression as c_expression

from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.platform.base_profiler import BaseProfiler


@PROFILERS.register_module(DeviceTarget.CPU.value)
class CpuProfiler(BaseProfiler):
    """
    CPU platform profiler
    """

    def __init__(self) -> None:
        super().__init__()
        self._prof_ctx = ProfilerContext()
        self._profiler = c_expression.Profiler.get_instance(DeviceTarget.CPU.value)

    def start(self) -> None:
        """Start profiling."""
        logger.info("CpuProfiler start.")
        self._profiler.init(self._prof_ctx.framework_path)
        logger.info("CpuProfiler framework_path: %s", self._prof_ctx.framework_path)
        self._profiler.step_profiling_enable(True)

        if ProfilerActivity.CPU in self._prof_ctx.activities:
            self._profiler.enable_op_time()

        if self._prof_ctx.profile_memory:
            self._profiler.enable_profile_memory()

    def stop(self) -> None:
        """Stop profiling."""
        logger.info("CpuProfiler stop.")
        self._profiler.stop()

    def analyse(self, **kwargs) -> None:
        """Analyse profiling data."""
        logger.info("CpuProfiler analyse.")

    def finalize(self) -> None:
        """Finalize profiling data."""
        logger.info("CpuProfiler finalize.")
