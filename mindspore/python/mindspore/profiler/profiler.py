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
from mindspore.profiler.common.registry import PROFILERS
from mindspore.profiler.common.constant import DeviceTarget
from mindspore.profiler.common.constant import ProfilerLevel
from mindspore.profiler.platform_profiler.prof_context import ProfContext


class NewProfiler:
    """
    Refactor profiler
    """

    def __init__(
            self,
            output_path: str = "./data",
            profiler_level: ProfilerLevel = None,
            op_time: bool = True,
            profile_communication: bool = False,
            profile_memory: bool = False,
            parallel_strategy: bool = False,
            start_profile: bool = True,
            aicore_metrics: int = 0,
            l2_cache: bool = False,
            hbm_ddr: bool = False,
            pcie: bool = False,
            sync_enable: bool = True,
            data_process: bool = False,
            timeline_limit: int = 500,
            profile_framework: str = None,
            with_stack: bool = False,
            data_simplification: bool = True,
            **kwargs) -> None:

        self._prof_context = ProfContext(
            output_path=output_path,
            profiler_level=profiler_level,
            op_time=op_time,
            profile_communication=profile_communication,
            profile_memory=profile_memory,
            parallel_strategy=parallel_strategy,
            start_profile=start_profile,
            aicore_metrics=aicore_metrics,
            l2_cache=l2_cache,
            hbm_ddr=hbm_ddr,
            pcie=pcie,
            sync_enable=sync_enable,
            data_process=data_process,
            timeline_limit=timeline_limit,
            profile_framework=profile_framework,
            with_stack=with_stack,
            data_simplification=data_simplification
        )

        self._has_started = False

        self._cpu_profiler = PROFILERS.get_modules().get(DeviceTarget.CPU.value)(
            op_time=self._prof_context.op_time,
            with_stack=self._prof_context.with_stack,
            data_process=self._prof_context.data_process,
            output_path=self._prof_context.output_path,
            profile_memory=self._prof_context.profile_memory,
            profile_framework=self._prof_context.profile_framework
        )

        self._device_target = self._prof_context.device_target
        self._device_profiler = PROFILERS.get_modules().get(self._device_target)(
            self._prof_context.get_args()
        )

    def start(self) -> None:
        """
        Used for Ascend, GPU, start profiling. Profiling can be turned on based on step and epoch.
        """
        if not self._has_started:
            self._has_started = True
        else:
            raise RuntimeError("The profiler has already started. Do not turn on again in the open state.")

        self._cpu_profiler.start()
        self._device_profiler.start()

    def stop(self) -> None:
        """
        Used for Ascend, GPU, stop profiling. Profiling can be turned off based on step and epoch.
        """
        if self._has_started:
            self._has_started = False
        else:
            raise RuntimeError("The profiler has not started, so can not stop. Please call the start() method "
                               "before calling the stop() method.")

        self._cpu_profiler.stop()
        self._device_profiler.stop()

    def analyse(self, offline_path=None, pretty=False, step_list=None, mode="sync") -> None:
        """
        Collect and analyze training performance data, support calls during and after training. The example shows above.

        Args:
            offline_path (Union[str, None], optional): The data path which need to be analyzed with offline mode.
                Offline mode isused in abnormal exit scenario. This parameter should be set to ``None``
                for online mode. Default: ``None``.
            pretty (bool, optional): Whether to pretty json files. Default: ``False``.
            step_list (list, optional): A list of steps that need to be analyzed. Default: ``None``.
                By default, all steps will be analyzed.
            mode (str, optional): Analysis mode, it must be one of ["sync", "async"]. Default: ``sync``.

                - sync: analyse data in current process, it will block the current process.
                - async: analyse data in subprocess, it will not the current process.Since the parsing process
                  will take up extra CPU resources, please enable this mode according to the actual resource situation.

        """
        self._cpu_profiler.stop(offline_path, pretty, step_list)
        self._device_profiler.stop(offline_path, pretty, step_list, mode)

    def op_analyse(self, op_name, device_id=None) -> None:
        """
        Profiler users can use this interface to obtain operator performance data.

        Args:
            op_name (str or list): The primitive operator name to query.
            device_id (int, optional): ID of the target device. This parameter is optional during network training or
                inference, and users can use device_id parameter to specify which card operator performance data to
                parse. If this interface is used for offline data parsing, Default: ``0`` .
        """

    @classmethod
    def offline_analyse(cls, path: str, pretty=False, step_list=None) -> None:
        """
        Analyze training performance data offline, which is invoked after performance data collection is completed.

        Args:
            path (str): The profiling data path which need to be analyzed offline.
                There needs to be a profiler directory in this path.
            pretty (bool, optional): Whether to pretty json files. Default: ``False``.
            step_list (list, optional): A list of steps that need to be analyzed. Default: ``None``.
                By default, all steps will be analyzed.
        """
        return
