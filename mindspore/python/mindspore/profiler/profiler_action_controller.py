# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""Profiler Action Controller"""
from functools import partial
from typing import Dict, Optional, Callable, Any

from mindspore.profiler.profiler_interface import ProfilerInterface
from mindspore.profiler.schedule import ProfilerAction

from mindspore import log as logger

__all__ = []


class ProfilerActionController:
    """
    A controller class for managing profiler actions and transitions.

    This class handles the actions and transitions between different profiler states.
    It uses two maps, abnormal_action_map and normal_action_map, to determine the actions
    to take based on the previous and current profiler actions.

    Attributes:
        profiler: The profiler instance associated with this controller.
        prof_interface (ProfilerInterface): The profiler interface instance.
        on_trace_ready (Optional[Callable[..., Any]]): A callback function to be called when the trace is ready.
        abnormal_action_map (Dict): A map of abnormal transitions and their corresponding actions.
        normal_action_map (Dict): A map of normal transitions and their corresponding actions.
    """

    def __init__(self, prof_interface: ProfilerInterface, on_trace_ready: Optional[Callable[..., Any]] = None) -> None:
        """
        Initializes a new instance of ProfActionController.

        Args:
            prof_interface (ProfilerInterface): The profiler interface instance.
            on_trace_ready (Optional[Callable[..., Any]]): A callback function to be called when the trace is ready.
        """
        self.prof_interface = prof_interface
        self.abnormal_action_map: Dict = self.init_abnormal_action_map()
        self.normal_action_map: Dict = self.init_normal_action_map()
        self.on_trace_ready = on_trace_ready

    def _trace_ready(self):
        """
        Calls the on_trace_ready callback function if it is set.

        This method is called when the trace is ready to notify the callback function.
        """
        if self.on_trace_ready:
            self.on_trace_ready()

    def transit_action(self, prev_action: ProfilerAction, current_action: ProfilerAction) -> None:
        """
        Handles actions between previous action and latter action

        Args:
            prev_action: The previous state
            current_action: the latter state
        """
        # Check whether the action is in the abnormal map
        action_list = self.abnormal_action_map.get((prev_action, current_action), [])
        if not action_list:
            if isinstance(prev_action, ProfilerAction) and isinstance(current_action, ProfilerAction):
                # Check whether the action is in the normal map
                action_list = self.handle_normal_action(prev_action, current_action)
        if not action_list:
            return
        for action in action_list:
            action()

    def handle_normal_action(self, start_state: ProfilerAction, end_state: ProfilerAction) -> list:
        """
        Obtain the action in the normal state

        Args:
            start_state: The previous state
            end_state: the latter state

        Returns:
            process_action list
        """
        process_action = []
        initial_state = start_state

        # Handle special case for ProfilerAction.RECORD_AND_SAVE to ProfilerAction.RECORD_AND_SAVE transition
        if start_state == ProfilerAction.RECORD_AND_SAVE and end_state == ProfilerAction.RECORD_AND_SAVE:
            process_action = [self.prof_interface.stop, self.prof_interface.finalize,
                              self._trace_ready, self.prof_interface.init,
                              self.prof_interface.start]
        else:
            while start_state != end_state:
                process_action.extend(self.normal_action_map[start_state])
                start_state = ProfilerAction.get_by_value((start_state.value + 1) % len(ProfilerAction))

            # Handle special cases for NONE to NONE, WARM_UP to WARM_UP, RECORD to RECORD transitions
            if initial_state == start_state and initial_state != ProfilerAction.RECORD_AND_SAVE:
                process_action = []

        return process_action

    def init_normal_action_map(self) -> dict:
        """
        Generate a normal action map

        Returns:
            normal_action_map map
        """
        return {
            ProfilerAction.NONE: [self.prof_interface.init],
            ProfilerAction.WARM_UP: [self.prof_interface.start],
            ProfilerAction.RECORD: [],
            ProfilerAction.RECORD_AND_SAVE: [
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear
            ]
        }

    def init_abnormal_action_map(self) -> dict:
        """
        Generate a abnormal action map

        Returns:
            abnormal_action_map map
        """
        return {
            (ProfilerAction.WARM_UP, ProfilerAction.NONE): [
                partial(logger.warning, "Incorrect schedule: WARMUP followed by NONE"),
                self.prof_interface.start,
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self.prof_interface.clear
            ],
            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(logger.warning, "Incorrect schedule: RECORD followed by NONE"),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self.prof_interface.clear
            ],
            (ProfilerAction.RECORD, ProfilerAction.WARM_UP): [
                partial(logger.warning, "Incorrect schedule: RECORD followed by WARMUP"),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self.prof_interface.clear
            ],
            # used for exit action
            (ProfilerAction.WARM_UP, None): [
                partial(logger.warning,
                        "Incorrect schedule: Stop profiler while current state is WARMUP "
                        "which will result in empty parsed data."),
                self.prof_interface.finalize,
                self.prof_interface.clear,
                self.prof_interface.delete_dir
            ],
            (ProfilerAction.RECORD, None): [
                partial(logger.warning,
                        "Incorrect schedule: Stop profiler while current state is RECORD "
                        "which may result in incomplete parsed data."),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear
            ],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                partial(logger.warning,
                        "Stop profiler while current state is RECORD_AND_SAVE, "
                        "perhaps the scheduling cycle has not yet completed."),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear
            ]
        }
