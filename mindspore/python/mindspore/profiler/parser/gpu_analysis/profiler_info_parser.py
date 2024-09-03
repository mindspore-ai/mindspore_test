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
"""Profiler host information parser"""
import os
import json
from decimal import Decimal
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.ascend_analysis.constant import Constant
from mindspore.profiler.parser.profiler_info import ProfilerInfo


class GPUProfilerInfoParser:
    """Parse files that record information, such as profiler_info.json"""

    _freq = 2600000000
    _system_time = 0
    _system_cnt = 0
    _s_to_ns = 1e9
    # profiler information related files
    _source_path = None
    _loaded_frequency = False
    _rank_id = 0

    @classmethod
    def init_source_path(cls, source_path: str):
        """initialize the path of PROF_* directory."""
        source_path = validate_and_normalize_path(source_path)
        cls._source_path = source_path

    @classmethod
    def init_rank_id(cls, rank_id: int):
        """initialize the rank id."""
        cls._rank_id = rank_id

    @classmethod
    def get_local_time(cls, syscnt: int) -> Decimal:
        """Convert syscnt to local time."""
        if not cls._loaded_frequency:
            profiler_info_path = os.path.join(cls._source_path, f"profiler_info_{cls._rank_id}.json")
            if not os.path.isfile(profiler_info_path):
                raise RuntimeError(f"Can`t find the file {profiler_info_path}, please check !")
            with os.fdopen(os.open(profiler_info_path, os.O_RDONLY, 0o600),
                           'r') as fr:
                profiler_info_data = json.load(fr)
            cls._system_cnt = profiler_info_data.get('system_cnt')
            cls._system_time = profiler_info_data.get('system_time')
            ProfilerInfo.set_system_time(cls._system_cnt)
            ProfilerInfo.set_system_cnt(cls._system_time)
            cls._loaded_frequency = True

        start_ns = cls._get_timestamp(syscnt)
        return Decimal(start_ns).quantize(Decimal('0.000')) * Decimal(Constant.NS_TO_US).quantize(Decimal('0.000'))

    @classmethod
    def _get_timestamp(cls, syscnt: int):
        """Convert syscnt to time stamp."""
        ratio = cls._freq / cls._s_to_ns
        # The unit of timestamp is ns
        timestamp = (syscnt - cls._system_cnt) / ratio + cls._system_time
        return timestamp
