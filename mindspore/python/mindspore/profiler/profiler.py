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
from typing import Optional as Opt

from mindspore.profiler.common.constant import ProfilerStepNameConstant
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager
from mindspore.profiler.common.record_function import RecordFunction
from mindspore.profiler.platform.npu_profiler import NPUProfilerAnalysis
from mindspore.profiler.profiler_action_controller import ProfilerActionController
from mindspore.profiler.profiler_interface import ProfilerInterface
from mindspore.profiler.schedule import _default_schedule_fn, ProfilerAction

from mindspore import log as logger


def tensor_board_trace_handler():
    try:
        NPUProfilerAnalysis.online_analyse()
        ProfilerPathManager.reset()
    except Exception as e:  # pylint: disable=W0703
        logger.error("Call tensorboard_trace_handler failed. Exception: %s", str(e))


class RefactorProfiler:
    """
    Refactor Profiler class
    """

    def __init__(self, **kwargs) -> None:
        self._prof_context: ProfilerContext = ProfilerContext(**kwargs)
        self._has_started: bool = False
        ProfilerInterface.init()
        profilerInterface = ProfilerInterface()
        self.schedule_arg = kwargs.get('schedule')
        if self.schedule_arg is not None:
            self.schedule = self._prof_context.schedule
            self._record_steps: bool = True
            self._schedule_no_use_step = True
        else:
            self.schedule = _default_schedule_fn
            self._record_steps: bool = False
            self._schedule_no_use_step = None
        self._step_rec_fn: Opt[RecordFunction] = None
        self.step_num = 0
        self.current_action: ProfilerAction = self.schedule(self.step_num)
        self.action_controller = ProfilerActionController(profilerInterface, self._prof_context.on_trace_ready)
        if self._prof_context.start_profile:
            self.start()

    def start(self) -> None:
        if not self._has_started:
            self._has_started = True
        else:
            logger.warning("The profiler has already started. Do not turn on again in the open state.")
            return
        self.action_controller.transit_action(ProfilerAction.NONE, self.current_action)
        if self._record_steps:
            self._step_rec_fn = RecordFunction(ProfilerStepNameConstant.PROFILER_STEP + str(self.step_num))
            self._step_rec_fn.start()

    def stop(self) -> None:
        """
        Stops the execution of the profiler.

        This method is responsible for halting the profile's operation and handling the associated logic.
        If the profiler has a schedule but has not used the `step()` method to collect data, a warning is logged.
        If the profiler has not been started, an error is logged. If the profiler has been started, it stops recording
        steps and updates the controller's state.

        Args:
            None

        Returns:
            None

        Raises:
            None

        Notes:
            - Ensure that the profiler has been properly initialized before calling this method.
            - If the profiler has not been started, calling this method will log an error.
        """
        if self._schedule_no_use_step:
            logger.warning("The profiler has schedule. Please use step() to collect data.")
            return
        if self._has_started:
            self._has_started = False
        else:
            logger.error("The profiler has not started. Do not turn off again in the closed state.")
            return
        if self._record_steps and self._step_rec_fn:
            self._step_rec_fn.stop()
        self.action_controller.transit_action(self.current_action, None)

    def analyse(self) -> None:
        ProfilerInterface.stop()
        ProfilerInterface.analyse()
        ProfilerInterface.finalize()

    def step(self) -> None:
        """
        Executes a single step in the profiling process.

        This method checks if the profiler is started and if a schedule is set. If both conditions are met,
        it proceeds to update the current action based on the schedule, records the transition using the
        action controller, and starts or stops recording as needed.

        The method's behavior is as follows:
        1. Checks if a schedule is set; if not, logs an error and returns.
        2. Checks if the profiler has started; if not, logs an error and returns.
        3. Stops the current recording function if recording steps are enabled.
        4. Updates the step number and calculates the new current action based on the schedule.
        5. Notifies the action controller of the action transition.
        6. Starts a new recording function if recording steps are enabled.

        Raises:
            None

        Returns:
            None
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

    def __enter__(self) -> None:
        self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def __del__(self):
        if self._has_started:
            self.stop()
