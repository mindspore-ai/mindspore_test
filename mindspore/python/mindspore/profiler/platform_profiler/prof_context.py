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
"""Profiler context"""
from mindspore import log as logger
from mindspore.profiler.common.singleton import Singleton


@Singleton
class ProfContext:
    """
    Profiler context.
    """
    def __init__(self, **kwargs):
        self._output_path = kwargs.get("output_path", "./data")
        self._profiler_level = kwargs.get("profiler_level", None)
        self._op_time = kwargs.get("op_time", True)
        self._profile_communication = kwargs.get("profile_communication", False)
        self._profile_memory = kwargs.get("profile_memory", False)
        self._parallel_strategy = kwargs.get("parallel_strategy", False)
        self._start_profile = kwargs.get("start_profile", True)
        self._aicore_metrics = kwargs.get("aicore_metrics", 0)
        self._l2_cache = kwargs.get("l2_cache", False)
        self._hbm_ddr = kwargs.get("hbm_ddr", False)
        self._pcie = kwargs.get("pcie", False)
        self._sync_enable = kwargs.get("sync_enable", True)
        self._data_process = kwargs.get("data_process", False)
        self._timeline_limit = kwargs.get("timeline_limit", 500)
        self._profile_framework = kwargs.get("profile_framework", None)
        self._with_stack = kwargs.get("with_stack", False)
        self._data_simplification = kwargs.get("data_simplification", True)

        self._dev_id = None
        self._rank_id = None
        self._device_target = None
        logger.info("ProfContext init success.")

    @property
    def output_path(self):
        return self._output_path

    @property
    def profiler_level(self):
        return self._profiler_level

    @property
    def op_time(self):
        return self._op_time

    @property
    def profile_communication(self):
        return self._profile_communication

    @property
    def profile_memory(self):
        return self._profile_memory

    @property
    def parallel_strategy(self):
        return self._parallel_strategy

    @property
    def start_profile(self):
        return self._start_profile

    @property
    def aicore_metrics(self):
        return self._aicore_metrics

    @property
    def l2_cache(self):
        return self._l2_cache

    @property
    def hbm_ddr(self):
        return self._hbm_ddr

    @property
    def pcie(self):
        return self._pcie

    @property
    def sync_enable(self):
        return self._sync_enable

    @property
    def data_process(self):
        return self._data_process

    @property
    def timeline_limit(self):
        return self._timeline_limit

    @property
    def profile_framework(self):
        return self._profile_framework

    @property
    def with_stack(self):
        return self._with_stack

    @property
    def data_simplification(self):
        return self._data_simplification

    @property
    def device_target(self):
        """Get device target."""
        return self._device_target

    @property
    def rank_id(self):
        """Get rank id."""
        return self._rank_id

    @property
    def dev_id(self):
        """Get device id."""
        return self._dev_id
