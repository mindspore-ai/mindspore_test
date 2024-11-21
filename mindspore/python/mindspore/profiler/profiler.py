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
import os
import json
from typing import Optional, Dict
from sys import getsizeof
from concurrent.futures import ProcessPoolExecutor, as_completed

from mindspore import log as logger
from mindspore.profiler.common.constant import ProfilerStepNameConstant
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.platform.npu_profiler import NPUProfilerAnalysis
from mindspore.profiler.profiler_action_controller import ProfilerActionController
from mindspore.profiler.profiler_interface import ProfilerInterface
from mindspore.profiler.schedule import _default_schedule_fn, ProfilerAction
from mindspore.profiler.common.record_function import RecordFunction
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager


def tensor_board_trace_handler():
    try:
        NPUProfilerAnalysis.online_analyse()
        if ProfilerContext().data_simplification:
            ProfilerPathManager().simplify_data()
    except Exception as e:  # pylint: disable=W0703
        logger.error("Call tensorboard_trace_handler failed. Exception: %s", str(e))


class Profiler:
    """
    Refactor Profiler class
    """
    MAX_META_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(self, **kwargs) -> None:
        self._metadata: Dict[str, str] = {}
        self._prof_context: ProfilerContext = ProfilerContext()
        self._prof_context.set_params(**kwargs)
        self._has_started: bool = False
        self.schedule_arg = kwargs.get('schedule')
        if self.schedule_arg is not None:
            self.schedule = self._prof_context.schedule
            self._record_steps: bool = True
            self._schedule_no_use_step = True
        else:
            self.schedule = _default_schedule_fn
            self._record_steps: bool = False
            self._schedule_no_use_step = None
        self._step_rec_fn: Optional[RecordFunction] = None
        self.step_num = 0
        self.current_action: ProfilerAction = self.schedule(self.step_num)
        self.action_controller = ProfilerActionController(ProfilerInterface, self._prof_context.on_trace_ready)
        if self._prof_context.start_profile:
            self.start()

    def start(self) -> None:
        """
        Start the profiler
        """
        if self._has_started:
            logger.warning("The profiler has already started. Do not turn on again in the open state.")
            return
        self._has_started = True
        self.action_controller.transit_action(ProfilerAction.NONE, self.current_action)
        if self._record_steps:
            self._step_rec_fn = RecordFunction(ProfilerStepNameConstant.PROFILER_STEP + str(self.step_num))
            self._step_rec_fn.start()

    def stop(self) -> None:
        """
        Stop the profiler
        """
        if self._schedule_no_use_step:
            logger.warning("The profiler has schedule. Please use step() to collect data.")
            return
        if not self._has_started:
            logger.error("The profiler has not started. Do not turn off again in the closed state.")
            return
        self._has_started = False
        if self._record_steps and self._step_rec_fn:
            self._step_rec_fn.stop()
        if self.schedule_arg:
            self.action_controller.transit_action(self.current_action, None)
        else:
            ProfilerInterface.stop()
        self._dump_metadata()

    def analyse(self, offline_path=None, pretty=False, step_list=None, mode="sync") -> None:
        """
        Analyse the profiling data.
        """
        if self._has_started:
            ProfilerInterface.stop()
            self._has_started = False

        if self.schedule_arg:
            logger.warning("The profiler has schedule. Please use 'on_trace_ready' to analyse data.")
            return

        ProfilerInterface.analyse()
        ProfilerInterface.finalize()

    @classmethod
    def offline_analyse(cls, path: str, pretty=False, step_list=None, data_simplification=False) -> None:
        """
        Analyze training performance data offline, which is invoked after performance data collection is completed.

        Args:
            path (str): The profiling data path which need to be analyzed offline.
                There needs to be a profiler directory in this path.
            pretty (bool, optional): Whether to pretty json files. Default: ``False``.
            step_list (list, optional): A list of steps that need to be analyzed. Default: ``None``.
                By default, all steps will be analyzed.
            data_simplification (bool, optional): Whether to enable data simplification. Default: ``True``.

        Examples:
            1. Single-device scenario:
            {path}
            ├── ASCEND_PROFILER_OUTPUT/
            ├── FRAMEWORK/
            └── PROF_{timestamp}/

            2. Multi-device scenario (rank 0 and rank 1):
            {path}
            ├── {}_ascend_ms/  # rank 0
            │   ├── ASCEND_PROFILER_OUTPUT/
            │   ├── FRAMEWORK/
            │   └── PROF_{}/
            └── {}_ascend_ms/  # rank 1

        >>> from mindspore import Profiler
        >>> Profiler.offline_analyse("./profiling_path")
        """
        real_path = PathManager.get_abs_path(path)
        PathManager.check_input_directory_path(real_path)
        ascend_ms_path_list = PathManager.get_ascend_ms_path_list(real_path)

        if not ascend_ms_path_list:
            msg = (f"Invalid path: {real_path}. Expected a *_ascend_ms_* directory "
                   "or a parent directory of multiple *_ascend_ms_*")
            logger.error(msg)
            return

        worker_number = min(os.cpu_count() // 2, len(ascend_ms_path_list))
        with ProcessPoolExecutor(max_workers=worker_number) as executor:
            futures = [
                executor.submit(
                    NPUProfilerAnalysis.offline_analyse,
                    ascend_ms_path,
                    pretty,
                    step_list,
                    data_simplification
                ) for ascend_ms_path in ascend_ms_path_list
            ]
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e: # pylint: disable=W0703
                    logger.error("offline analysis failed: %s", str(e))

    def step(self) -> None:
        """
        Step the profiler.
        """
        if self.schedule_arg is None:
            logger.error("With no schedule in the Profiler, step takes no effect!")
            return
        if not self._has_started:
            logger.error("Profiler is stopped, step takes no effect!")
            return
        if self._step_rec_fn:
            self._step_rec_fn.stop()
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)
        self.action_controller.transit_action(prev_action, self.current_action)
        self._step_rec_fn = RecordFunction(ProfilerStepNameConstant.PROFILER_STEP + str(self.step_num))
        self._step_rec_fn.start()
        self._schedule_no_use_step = False

    def add_metadata(self, key: str, value: str):
        """
        Report custom metadata key-value pair data.

        Args:
            key (str): The key to the metadata.
            value (str): The value to the metadata.

        Examples:
            >>> from mindspore import Profiler
            >>> # Profiler init.
            >>> profiler = Profiler()
            >>> # Call Profiler add_metadata
            >>> profiler.add_metadata("test_key", "test_value")
            >>> # Profiler end
            >>> profiler.analyse()
        """
        if not isinstance(key, str) or not isinstance(value, str):
            logger.warning("The key and value of metadata must be string. Skip this metadata.")
            return

        add_size = getsizeof(key) + getsizeof(value)
        if getsizeof(self._metadata) + add_size < self.MAX_META_SIZE:
            if key in self._metadata:
                logger.warning(f"{key} is already saved as metadata, override it.")
            self._metadata[key] = value
        else:
            logger.warning("Too many metadata added. Skip this metadata")

    def add_metadata_json(self, key: str, value: str):
        """
        Report custom metadata key-value pair data with the value as a JSON string data.

        Args:
            key (str): The key to the metadata.
            value (str): The json str format value to the metadata.

        Examples:
            >>> import json
            >>> from mindspore import Profiler
            >>> # Profiler init.
            >>> profiler = Profiler()
            >>> # Call Profiler add_metadata_json
            >>> profiler.add_metadata_json("test_key", json.dumps({"key1": 1, "key2": 2}))
            >>> # Profiler end, metadata will be saved in profiler_metadata.json
            >>> profiler.analyse()
        """
        if not isinstance(key, str) or not isinstance(value, str):
            logger.warning("The key and value of metadata must be string. Skip this metadata.")
            return

        add_size = getsizeof(key) + getsizeof(value)
        if getsizeof(self._metadata) + add_size < self.MAX_META_SIZE:
            try:
                if key in self._metadata:
                    logger.warning(f"{key} is already saved as metadata, override it.")
                self._metadata[key] = json.loads(value)
            except ValueError:
                logger.warning("The metadata value must be json format string. Skip this metadata")
        else:
            logger.warning("Too many metadata added. Skip this metadata")

    def _dump_metadata(self):
        """Dump metadata to file."""
        if not self._metadata:
            return
        save_path = os.path.join(self._prof_context.ascend_ms_dir, "profiler_metadata.json")
        FileManager.create_json_file(save_path, self._metadata, indent=4)
        self._metadata.clear()

    def __enter__(self) -> 'Profiler':
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def __del__(self):
        if self._has_started:
            self.stop()
            logger.warning("Profiler is stopped at the end of the program.")
