# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Record profiler information"""
import glob
import os
import stat

from mindspore.version import __version__ as ms_version
from mindspore.profiler.parser.ascend_analysis.file_manager import FileManager
from mindspore import log as logger


class ProfilerInfo:
    """
    This class is used to record profiler information.
    it contains context_mode, rank_id, rank_size, parallel_mode, pipeline_stage_num, pipeline_stage_id,
    profiling_start_time, profiling_stop_time, analyse_start_time, analyse_end_time
    """

    _file_name = "profiler_info_{}.json"
    _file_path = ""
    _profiler_info_dict = dict()
    JIT_LEVEL = "jit_level"

    @staticmethod
    def init_info(context_mode, rank_id):
        """Profiler info initialization must include context_mode, rank_id and output_path."""
        ProfilerInfo._profiler_info_dict["context_mode"] = context_mode
        ProfilerInfo._profiler_info_dict["rank_id"] = rank_id
        ProfilerInfo._profiler_info_dict["ms_version"] = ms_version
        ProfilerInfo._file_name = ProfilerInfo._file_name.format(rank_id)

    @staticmethod
    def set_parallel_info(parallel_mode="", stage_num=1):
        """Set parallel info include parallel_mode, pipeline_stage_num and pipeline_stage_id."""
        info = dict()
        info["parallel_mode"] = parallel_mode
        info["stage_num"] = stage_num
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_profiling_start_time(start_time):
        """Set the profiling start time."""
        info = dict()
        info["profiling_start_time"] = start_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_profiling_stop_time(stop_time):
        """Set the profiling stop time."""
        info = dict()
        info["profiling_stop_time"] = stop_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_analyse_start_time(start_time):
        """Set the analyse start time."""
        info = dict()
        info["analyse_start_time"] = start_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_analyse_end_time(end_time):
        """Set the analyse end time."""
        info = dict()
        info["analyse_end_time"] = end_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_export_start_time(start_time):
        """Set the export start time."""
        info = dict()
        info["export_start_time"] = start_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_export_end_time(end_time):
        """Set the export end time."""
        info = dict()
        info["export_end_time"] = end_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_export_flag(flag):
        """Set whether all-export or not."""
        ProfilerInfo._profiler_info_dict["all_export"] = flag

    @staticmethod
    def set_system_time(sys_time):
        """Set system time."""
        ProfilerInfo._profiler_info_dict["system_time"] = sys_time

    @staticmethod
    def set_system_cnt(sys_cnt):
        """Set system cnt."""
        ProfilerInfo._profiler_info_dict["system_cnt"] = sys_cnt

    @staticmethod
    def set_diff_time(diff_time):
        """synchronize timestamps between different devices"""
        ProfilerInfo._profiler_info_dict["diff_time"] = diff_time

    @staticmethod
    def set_graph_ids(graph_ids):
        """Set the graph id list."""
        ProfilerInfo._profiler_info_dict["graph_ids"] = graph_ids

    @staticmethod
    def set_rank_size(rank_size):
        """Set the rank size."""
        ProfilerInfo._profiler_info_dict["rank_size"] = rank_size

    @staticmethod
    def set_heterogeneous(is_heterogeneous):
        """Set is it heterogeneous."""
        ProfilerInfo._profiler_info_dict["is_heterogeneous"] = is_heterogeneous

    @staticmethod
    def get_profiler_info():
        """Get the profiler info."""
        return ProfilerInfo._profiler_info_dict

    @staticmethod
    def set_profiling_options(profiling_options):
        """Set profiling options to profiler info dict"""
        ProfilerInfo._profiler_info_dict["profiling_options"] = profiling_options

    @staticmethod
    def set_jit_level(jit_level):
        """Set jit_level to profiler info dict"""
        ProfilerInfo._profiler_info_dict[ProfilerInfo.JIT_LEVEL] = jit_level

    @staticmethod
    def set_data_simplification(data_simplification):
        """
        Function Description:
            Set the data simplification to profiler info dict
        Parameter:
            data_simplification: Whether data simplification is enabled
        """
        ProfilerInfo._profiler_info_dict["data_simplification"] = data_simplification

    @staticmethod
    def save(output_path):
        """Save the profiler info to file."""
        ProfilerInfo._file_path = os.path.join(output_path, ProfilerInfo._file_name)
        FileManager.create_json_file(output_path, ProfilerInfo._profiler_info_dict, ProfilerInfo._file_name, indent=4)
        os.chmod(ProfilerInfo._file_path, stat.S_IREAD | stat.S_IWRITE)

    @staticmethod
    def load_profiler_info_dict(input_path):
        """Load the profiler info from input path."""
        ProfilerInfo._file_path = os.path.join(input_path, ProfilerInfo._file_name)
        try:
            load_info_dict = FileManager.read_json_file(ProfilerInfo._file_path)
        except RuntimeError as err:
            logger.warning(f"Cannot read file: {ProfilerInfo._file_path}, Error: {err}")
            return
        if not load_info_dict:
            msg = f"Offline analysis failed load the ProfilerInfo._profiler_info_dict from: {ProfilerInfo._file_path}"
            logger.warning(msg)
            return
        ProfilerInfo._profiler_info_dict = load_info_dict
        os.chmod(ProfilerInfo._file_path, stat.S_IREAD | stat.S_IWRITE)

    @staticmethod
    def get_rank_id(profiler_dir: str):
        """
        Function Description:
            Get rank id from profiler_info_*.json
        Parameter:
            profiler_dir: the directory path of profiler data, eg: rank_0/profiler
        Return:
            str type rank id
        """
        prof_info_path = os.path.join(profiler_dir, "profiler_info_*.json")
        prof_info_path = glob.glob(prof_info_path)
        if not prof_info_path:
            logger.warning("Cannot find profiler_info.json in the profiler directory.")
            return "-1"

        info_data = FileManager.read_json_file(prof_info_path[0])
        return info_data.get("rank_id", "-1")

    @staticmethod
    def get_device_id(prof_dir: str):
        """
        Function Description:
            Get device id from PROF_XXX dir
        Parameter:
            prof_dir: the directory path of PROF_XXX
        Return:
            str type device id
        """
        device_dir = os.path.join(prof_dir, "device_*")
        device_dir = glob.glob(device_dir)
        if not device_dir:
            logger.warning("Cannot find device_XXX in the %s.", prof_dir)
            return "-1"

        return device_dir[0].split("device_")[-1]
