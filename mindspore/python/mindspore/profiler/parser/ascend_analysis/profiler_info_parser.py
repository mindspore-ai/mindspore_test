# Copyright 2023 Huawei Technologies Co., Ltd
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
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import Popen, PIPE
from configparser import ConfigParser

from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.ascend_analysis.constant import Constant


class ProfilerInfoParser:
    """Parse files that record information, such as profiler_info.json"""

    _freq = 100.0
    _msprof_cmd = "msprof"
    _time_out = 1
    # profiler information related files
    _source_prof_path = None
    _loaded_frequency = False
    _rank_id = 0
    _clock_monotonic_raw = 0
    _cntvct = 0
    _collectionTimeBegin = 0
    _clockMonotonicRaw = 0
    _get_localtime_diff = 0

    @classmethod
    def init_source_path(cls, source_path: str):
        """initialize the path of PROF_* directory."""
        source_path = validate_and_normalize_path(source_path)
        prof_path = os.path.dirname(source_path)
        dir_name = os.path.basename(source_path)
        if not dir_name.startswith("device") or not os.path.exists(source_path):
            raise RuntimeError("Input source path is invalid!")
        cls._source_prof_path = prof_path

    @classmethod
    def init_rank_id(cls, rank_id: int):
        """initialize the rank id."""
        cls._rank_id = rank_id

    @classmethod
    def get_local_time(cls, syscnt: int) -> Decimal:
        """Convert syscnt to local time."""
        if not cls._loaded_frequency:
            outs, _ = cls.__run_cmd(['which', cls._msprof_cmd])
            if not outs:
                raise FileNotFoundError("Failed to find msprof command!")
            msprof_path = os.path.realpath(outs.strip())
            sup_path = msprof_path.split('tools')[0]
            script_path = os.path.join(sup_path, 'tools/profiler/profiler_tool/analysis/interface/get_msprof_info.py')
            py_cmd = ['python', script_path, '-dir', os.path.join(cls._source_prof_path, 'host')]
            outs, _ = cls.__run_cmd(py_cmd)
            if not outs:
                raise RuntimeError("Failed to get msprof information!")
            result = json.loads(outs)
            cpu_info = result.get('data', {}).get('host_info', {}).get('cpu_info', [{}])[0]
            try:
                cls._freq = float(cpu_info.get("Frequency", cls._freq))
            except ValueError:
                pass
            cls._get_msprof_timestamp(cls._source_prof_path)
            cls._loaded_frequency = True
        start_ns = cls.__get_timestamp(syscnt)
        return Decimal(start_ns).quantize(Decimal('0.000')) * Decimal(Constant.NS_TO_US).quantize(Decimal('0.000'))

    @classmethod
    def _get_msprof_timestamp(cls, source_path):
        """get msprof timestamp info"""
        start_log = ConfigParser()
        start_log.read(os.path.join(source_path, "host", "host_start.log"))
        cls._clock_monotonic_raw = int(start_log.get("Host", "clock_monotonic_raw"))
        cls._cntvct = int(start_log.get("Host", "cntvct"))

        with open(os.path.join(source_path, "host", "start_info"), "r") as f:
            info_dict = json.load(f)
            cls._collectionTimeBegin = int(info_dict.get("collectionTimeBegin", 0))  # us
            cls._clockMonotonicRaw = int(info_dict.get("clockMonotonicRaw", 0))
            us_to_ns = 1000
            cls._get_localtime_diff = cls._clock_monotonic_raw + (cls._collectionTimeBegin * us_to_ns -
                                                                  cls._clockMonotonicRaw)

    @classmethod
    def __run_cmd(cls, cmd):
        """run shell command"""
        try:
            with Popen(cmd, stdout=PIPE, stderr=PIPE, text=True) as proc:
                outs, errs = proc.communicate(timeout=cls._time_out)
        except (FileNotFoundError, PermissionError, CalledProcessError) as exc:
            raise RuntimeError(exc) from exc
        except TimeoutExpired as err:
            proc.kill()
            msg = "The possible cause is that too much data is collected " \
                  "and the export time is too long."
            logger.error(msg)
            raise TimeoutError(msg) from err
        logger.info(outs)
        return outs, errs

    @classmethod
    def __get_timestamp(cls, syscnt: int, time_fmt: int = 1000):
        """Convert syscnt to time stamp."""
        ratio = time_fmt / cls._freq
        # The unit of timestamp is ns
        timestamp = round((syscnt - cls._cntvct) * ratio) + cls._get_localtime_diff
        return timestamp
