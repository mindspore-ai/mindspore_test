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
"""Profiling api file."""
from mindspore import log as logger
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.profiler_interface import ProfilerInterface


class RefactorProfiler:
    """
    Refactor Profiler class
    """
    def __init__(self, **kwargs) -> None:
        self._prof_context: ProfilerContext = ProfilerContext(**kwargs)
        self._has_started: bool = False
        ProfilerInterface.init()
        if self._prof_context.start_profile:
            self.start()

    def start(self) -> None:
        if not self._has_started:
            self._has_started = True
        else:
            logger.error("Profiler has already started, do not start again.")
            return
        ProfilerInterface.init()
        ProfilerInterface.start()

    def stop(self) -> None:
        if self._has_started:
            self._has_started = False
        else:
            logger.error("Profiler has not started, start it first.")
            return
        ProfilerInterface.stop()

    def analyse(self) -> None:
        ProfilerInterface.analyse()
        ProfilerInterface.finalize()

    def step(self) -> None:
        pass

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def __del__(self):
        pass
