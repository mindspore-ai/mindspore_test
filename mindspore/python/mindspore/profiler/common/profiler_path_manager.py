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
"""Profiler Path Manager"""
import os
import socket
from datetime import datetime, timezone

from mindspore import log as logger
from mindspore.profiler.common.singleton import Singleton
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.common.path_manager import PathManager


@Singleton
class ProfilerPathManager:
    """
    ProfilerPathManager is responsible for creating and managing all paths used by profiler.
    """

    _ASCEND_MS_DIR = "{}_{}_{}_ascend_ms"

    def __init__(self):
        self._prof_ctx = ProfilerContext()

    def clean_analysis_cache(self):
        """
        Clean the profiler analysis cache.
        """
        ANALYSIS_CACHE = (
            # ASEND_PROFILER_OUTPUT_PATH
            self._prof_ctx.ascend_profiler_output_path,
            # PROF_XXX/mindstudio_profiler_output
            self._prof_ctx.msprof_profile_output_path,
            # PROF_XXX/mindstudio_profiler_log
            self._prof_ctx.msprof_profile_log_path,
            # PROF_XXX/host/sqlite
            os.path.join(self._prof_ctx.msprof_profile_host_path, "sqlite"),
            # PROF_XXX/host/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_host_path, "data", "all_file.complete"),
            # PROF_XXX/device_x/sqlite
            os.path.join(self._prof_ctx.msprof_profile_device_path, "sqlite"),
            # PROF_XXX/device_x/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_device_path, "data", "all_file.complete"),
        )

        for cache_path in ANALYSIS_CACHE:
            if os.path.isfile(cache_path):
                PathManager.remove_file_safety(cache_path)
            elif os.path.isdir(cache_path):
                PathManager.remove_path_safety(cache_path)

    def simplify_data(self):
        """
        Simplify the profiler data.
        """
        SIMPLIFY_CACHE = (
            # PROF_XXX/mindstudio_profiler_output
            self._prof_ctx.msprof_profile_output_path,
            # PROF_XXX/mindstudio_profiler_log
            self._prof_ctx.msprof_profile_log_path,
            # PROF_XXX/host/sqlite
            os.path.join(self._prof_ctx.msprof_profile_host_path, "sqlite"),
            # PROF_XXX/host/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_host_path, "data", "all_file.complete"),
            # PROF_XXX/device_x/sqlite
            os.path.join(self._prof_ctx.msprof_profile_device_path, "sqlite"),
            # PROF_XXX/device_x/data/all_file.complete
            os.path.join(self._prof_ctx.msprof_profile_device_path, "data", "all_file.complete"),
        )

        for cache_path in SIMPLIFY_CACHE:
            if os.path.isfile(cache_path):
                PathManager.remove_file_safety(cache_path)
            elif os.path.isdir(cache_path):
                PathManager.remove_path_safety(cache_path)

    def create_output_path(self):
        """
        Create ASCEND_PROFILER_OUTPUT dir, this method should call before analysis
        """
        PathManager.make_dir_safety(self._prof_ctx.ascend_profiler_output_path)

    def set_ascend_ms_dir(self):
        """
        reset xxx_ascend_ms name
        """
        self._prof_ctx.ascend_ms_dir = self._get_ascend_ms_dir()

    def create_profiler_paths(self):
        """
        Create xxx_ascend_ms and FRAMEWORK dir, this method should call before Profiler start
        """
        PathManager.make_dir_safety(self._prof_ctx.ascend_ms_dir)
        PathManager.make_dir_safety(self._prof_ctx.framework_path)
        logger.info(
            "Profiler ascend_ms_dir initialized: %s", self._prof_ctx.ascend_ms_dir
        )

    def _get_ascend_ms_dir(self) -> str:
        """
        Generate xxx_ascend_ms name
        """
        return self._ASCEND_MS_DIR.format(
            socket.gethostname(),
            os.getpid(),
            # save time with microseconds
            datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3],
        )
