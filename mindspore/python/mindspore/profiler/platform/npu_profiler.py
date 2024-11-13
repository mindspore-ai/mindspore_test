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
"""NPU platform profiler."""
import os
import re
import glob
import json
from typing import List, Optional

from mindspore import context
from mindspore import log as logger
import mindspore._c_dataengine as cde
import mindspore._c_expression as c_expression
from mindspore.profiler.common.registry import PROFILERS
from mindspore.profiler.common.constant import DeviceTarget, ProfilerActivity

from mindspore._c_expression import _framework_profiler_enable_mi
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.platform.base_profiler import BaseProfiler
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager
from mindspore.profiler.common.profiler_info import ProfilerInfo
from mindspore.profiler.analysis.task_manager import TaskManager
from mindspore.profiler.analysis.time_converter import TimeConverter
from mindspore.profiler.analysis.parser.ascend_cann_parser import AscendMsprofParser
from mindspore.profiler.analysis.parser.base_parser import DummyParser
from mindspore.profiler.analysis.parser.ms_framework_parser import FrameworkParser
from mindspore.profiler.analysis.parser.ms_minddata_parser import MindDataParser
from mindspore.profiler.analysis.parser.framework_cann_relation_parser import FrameworkCannRelationParser
from mindspore.profiler.analysis.viewer.ms_dataset_viewer import MsDatasetViewer
from mindspore.profiler.analysis.viewer.ascend_timeline_viewer import AscendTimelineViewer
from mindspore.profiler.analysis.viewer.ascend_kernel_details_viewer import AscendKernelDetailsViewer
from mindspore.profiler.analysis.viewer.ascend_step_trace_time_viewer import AscendStepTraceTimeViewer
from mindspore.profiler.analysis.viewer.ascend_integrate_viewer import AscendIntegrateViewer
from mindspore.profiler.analysis.viewer.ascend_memory_viewer import AscendMemoryViewer
from mindspore.profiler.analysis.viewer.ms_minddata_viewer import (
    MindDataPipelineRawViewer,
    MindDataPiplineSummaryViewer,
)


@PROFILERS.register_module(DeviceTarget.NPU.value)
class NpuProfiler(BaseProfiler):
    """
    NPU platform profiler
    """

    def __init__(self) -> None:
        super().__init__()
        self._prof_ctx = ProfilerContext()
        self._prof_info = ProfilerInfo()
        self._prof_path_mgr = ProfilerPathManager()
        self._prof_path_mgr.set_ascend_ms_dir()
        self._profiler = None
        if ProfilerActivity.NPU in self._prof_ctx.activities:
            self._profiler = c_expression.Profiler.get_instance(
                DeviceTarget.NPU.value
            )
            # initialize profiler backend
            self._profiler.init(
                self._prof_ctx.ascend_ms_dir,
                int(self._prof_ctx.device_id),
                json.dumps(self._prof_ctx.npu_profiler_params),
            )

        # record original profiler params
        self._prof_info.profiler_parameters = self._prof_ctx.original_params
        self._prof_info.ms_profiler_info = {
            "context_mode": context.get_context("mode"),
            "rank_id": self._prof_ctx.rank_id,
        }

        # initialize minddata profiler
        if self._prof_ctx.data_process:
            self._md_profiler = cde.GlobalContext.profiling_manager()
            self._md_profiler.init()
            logger.info("NpuProfiler init minddata profiler")

        self._prof_mgr = c_expression.ProfilerManager.get_instance()

    def start(self) -> None:
        """Start profiling."""
        logger.info("NpuProfiler start.")
        self._prof_path_mgr.create_profiler_paths()

        if ProfilerActivity.CPU in self._prof_ctx.activities:
            _framework_profiler_enable_mi()
            self._prof_mgr.set_profile_framework("time")
            logger.info("NpuProfiler start enable framework")

        if self._profiler:
            self._profiler.start()

        if self._prof_ctx.data_process:
            self._md_profiler.start()

    def stop(self) -> None:
        """Stop profiling."""
        logger.info("NpuProfiler stop.")
        if self._profiler:
            self._profiler.stop()

        if ProfilerActivity.CPU in self._prof_ctx.activities:
            self._prof_mgr.set_profile_framework("NULL")
            logger.info("NpuProfiler stop disable framework")

        if self._prof_ctx.data_process:
            self._md_profiler.stop()
            self._md_profiler.save(self._prof_ctx.framework_path)

        self._prof_info.save(self._prof_ctx.ascend_ms_dir, self._prof_ctx.rank_id)

    def analyse(self, **kwargs) -> None:
        """Analyse the profiling data."""
        logger.info("NpuProfiler analyse.")

        NPUProfilerAnalysis.online_analyse()
        if self._prof_ctx.data_simplification:
            self._prof_path_mgr.simplify_data()

    def finalize(self) -> None:
        """Finalize profiling data."""
        logger.info("NpuProfiler finalize.")
        if self._profiler:
            self._profiler.finalize()


class NPUProfilerAnalysis:
    """
    NPU profiler analysis interface
    """
    @classmethod
    def online_analyse(cls):
        """
        Online analysis for NPU
        """
        cls._pre_analyse_online()
        cls._run_tasks(**ProfilerContext().to_dict())

    @classmethod
    def offline_analyse(cls, path: str, pretty: bool,
                        step_list: Optional[List[int]], data_simplification: bool) -> None:
        """Analyze profiling data in offline mode."""
        cls._pre_analyse_offline(path, pretty, step_list, data_simplification)
        cls._run_tasks(**ProfilerContext().to_dict())

    @classmethod
    def _pre_analyse_online(cls):
        """
        Pre-process for online analysis
        """
        prof_ctx = ProfilerContext()
        prof_dir = glob.glob(os.path.join(prof_ctx.ascend_ms_dir, "PROF_*"))
        if not prof_dir:
            logger.error(f"No PROF_* directory found in {prof_ctx.ascend_ms_dir}")
            return

        prof_ctx.msprof_profile_path = prof_dir[0]
        ProfilerPathManager().clean_analysis_cache()
        ProfilerPathManager().create_output_path()
        ProfilerInfo().load_time_parameters(
            prof_ctx.msprof_profile_path,
            prof_ctx.msprof_profile_host_path
        )
        TimeConverter.init_parameters(**ProfilerInfo().time_parameters)

    @classmethod
    def _pre_analyse_offline(cls, ascend_ms_dir: str, pretty: bool, step_list: Optional[List[int]],
                             data_simplification: bool) -> None:
        """Pre-process profiling data for offline analysis."""
        prof_dir = glob.glob(os.path.join(ascend_ms_dir, "PROF_*"))
        if not prof_dir:
            logger.error(f"No PROF_* directory found in {ascend_ms_dir}")
            return

        # get device_id and rank_id from ascend_ms_dir
        device_id = cls.get_device_id(prof_dir[0])
        rank_id = cls.get_rank_id(ascend_ms_dir)

        prof_ctx = ProfilerContext()
        prof_ctx.set_params()
        prof_path_mgr = ProfilerPathManager()

        prof_info = ProfilerInfo()

        # set PROF_XXX path
        prof_ctx.ascend_ms_dir = ascend_ms_dir
        prof_ctx.msprof_profile_path = prof_dir[0]
        prof_info.load_time_parameters(
            prof_ctx.msprof_profile_path,
            prof_ctx.msprof_profile_host_path
        )
        prof_ctx.rank_id = rank_id
        prof_ctx.device_id = device_id
        prof_ctx.pretty = pretty
        prof_ctx.step_list = step_list
        prof_ctx.data_simplification = data_simplification
        prof_path_mgr.clean_analysis_cache()
        prof_path_mgr.create_output_path()

        prof_info.load_info(ascend_ms_dir, prof_ctx.rank_id)
        prof_ctx.load_offline_profiler_params(prof_info.profiler_parameters)
        TimeConverter.init_parameters(**prof_info.time_parameters)

    @classmethod
    def _run_tasks(cls, **kwargs) -> None:
        """
        Run tasks for online analysis
        """
        task_mgr = TaskManager()
        task_mgr.create_flow(
            AscendMsprofParser(**kwargs),
            FrameworkParser(**kwargs).register_post_hook(
                MsDatasetViewer(**kwargs).save
            ),
            FrameworkCannRelationParser()
            .register_post_hook(AscendTimelineViewer(**kwargs).save)
            .register_post_hook(AscendKernelDetailsViewer(**kwargs).save)
            .register_post_hook(AscendStepTraceTimeViewer(**kwargs).save)
            .register_post_hook(AscendIntegrateViewer(**kwargs).save),
            flow_name="cann_flow",
            show_process=True,
        )

        enable_data_process = kwargs.get("data_process", False)
        if enable_data_process:
            task_mgr.create_flow(
                MindDataParser(**kwargs)
                .register_post_hook(MindDataPipelineRawViewer(**kwargs).save)
                .register_post_hook(MindDataPiplineSummaryViewer(**kwargs).save),
                flow_name="minddata_flow",
                show_process=False,
            )

        enable_profile_memory = kwargs.get("profile_memory", False)
        if enable_profile_memory:
            task_mgr.create_flow(
                DummyParser().register_post_hook(AscendMemoryViewer(**kwargs).save),
                flow_name="memory_flow",
                show_process=False,
            )
        task_mgr.run()
        logger.info(json.dumps(task_mgr.cost_time, indent=4))

    @staticmethod
    def get_rank_id(ascend_ms_dir: str) -> str:
        """
        Function Description:
            Get rank id from profiler_info_*.json
        Parameter:
            ascend_ms_dir: the directory path of profiler data, eg: xxx_ascend_ms
        Return:
            str type rank id
        """
        prof_info_path = os.path.join(ascend_ms_dir, "profiler_info_*.json")
        prof_info_path = glob.glob(prof_info_path)
        if not prof_info_path:
            raise ValueError(f"Cannot find profiler_info.json in the {ascend_ms_dir}")

        pattern = r"profiler_info_(\d+)\.json$"
        match = re.search(pattern, prof_info_path[0])
        rank_id = match.group(1) if match else None

        if rank_id is None:
            raise ValueError(f"Cannot find rank id in {prof_info_path[0]}")

        return rank_id

    @staticmethod
    def get_device_id(prof_dir: str) -> str:
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
            raise ValueError(f"Cannot find device_* directory in the {prof_dir}")

        pattern = r"device_(\d+)$"
        match = re.search(pattern, device_dir[0])
        device_id = match.group(1) if match else None

        if device_id is None:
            raise ValueError(f"Cannot find device id in {device_dir[0]}")

        return device_id
