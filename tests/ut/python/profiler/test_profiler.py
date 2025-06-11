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
"""Test the dynamic profiler differentiation step."""
import unittest
from unittest.mock import patch

from mindspore.profiler.profiler import Profiler
from mindspore.profiler.profiler_action_controller import ProfilerActionController
from mindspore.profiler.profiler_interface import ProfilerInterface
from mindspore.profiler.schedule import Schedule, ProfilerAction


# pylint: disable=protected-access
class TestProfiler(unittest.TestCase):
    """Test the dynamic profiler differentiation step."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.init")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.start")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.stop")
    def test_should_profiler_start_stop_when_no_schedule(self, mock_prof_interface_stop, mock_prof_interface_start,
                                                         mock_prof_interface_init):
        """
            Turn off the start_profiler switch and schedule switch, verify the start---->stop functions.
        """
        # Construct the Profiler instance
        profiler = Profiler(start_profile=False)

        # The first time the start method is called
        profiler.start()
        self.assertTrue(profiler._has_started)
        mock_prof_interface_init.assert_called_once()
        mock_prof_interface_start.assert_called_once()
        self.assertEqual(profiler.current_action, ProfilerAction.RECORD)

        # The first time the stop method is called
        profiler.stop()
        self.assertEqual(profiler._has_started, False)
        self.assertEqual(profiler.current_action, ProfilerAction.RECORD)
        mock_prof_interface_stop.assert_called_once()

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.init")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.start")
    def test_should_profiler_start_start_when_no_schedule(self, mock_prof_interface_start, mock_prof_interface_init,
                                                          mock_logger_warning):
        """
            Turn off the start_profiler switch and schedule switch, verify the start---->start functions.
        """
        # Construct the Profiler instance
        profiler = Profiler(start_profile=False)

        # The first time the start method is called
        profiler.start()
        self.assertTrue(profiler._has_started)
        mock_prof_interface_init.assert_called_once()
        mock_prof_interface_start.assert_called_once()
        self.assertEqual(profiler.current_action, ProfilerAction.RECORD)

        # The second time the start method is called, the error log should be displayed
        profiler.start()
        mock_logger_warning.assert_called_with("The profiler has already started. Do not turn on again in the "
                                               "open state.")

    @patch("mindspore.log.error")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.init")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.start")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.stop")
    def test_should_profiler_start_stop_stop_when_no_schedule(self, mock_prof_interface_stop, mock_prof_interface_start,
                                                              mock_prof_interface_init, mock_logger_error):
        """
            Turn off the start_profiler switch and schedule switch, verify the start--->stop---->stop functions.
        """
        # Construct the Profiler instance
        profiler = Profiler(start_profile=False)

        # The first time the start method is called
        profiler.start()
        self.assertTrue(profiler._has_started)
        mock_prof_interface_init.assert_called_once()
        mock_prof_interface_start.assert_called_once()
        self.assertEqual(profiler.current_action, ProfilerAction.RECORD)

        # The first time the stop method is called, the error log should be displayed
        profiler.stop()
        self.assertEqual(profiler._has_started, False)
        self.assertEqual(profiler.current_action, ProfilerAction.RECORD)
        mock_prof_interface_stop.assert_called_once()

        # The second time the stop method is called, the error log should be displayed
        profiler.stop()
        mock_logger_error.assert_called_with("The profiler has not started. Do not turn off again in the closed "
                                             "state.")

    @patch("mindspore.log.error")
    def test_should_profiler_no_start_stop_when_no_schedule(self, mock_logger_error):
        """
            Turn off the start_profiler switch and schedule switch, verify the no_start--->stop functions.
        """
        # Construct the Profiler instance
        profiler = Profiler(start_profile=False)

        # The first time the stop method is called, the error log should be displayed
        profiler.stop()
        mock_logger_error.assert_called_with("The profiler has not started. Do not turn off again in the closed "
                                             "state.")

    @patch("mindspore.log.error")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.init")
    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.start")
    def test_should_profiler_stop_start_when_no_schedule(self, mock_prof_interface_start, mock_prof_interface_init,
                                                         mock_logger_error):
        """
            Turn off the start_profiler switch and schedule switch, verify the stop---->start functions.
        """
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False)

        # The first time the stop method is called, the error log should be displayed
        profiler.stop()
        mock_logger_error.assert_called_with("The profiler has not started. Do not turn off again in the closed "
                                             "state.")

        # The first time the start method is called
        profiler.start()
        self.assertTrue(profiler._has_started)
        mock_prof_interface_init.assert_called_once()
        mock_prof_interface_start.assert_called_once()
        self.assertEqual(profiler.current_action, ProfilerAction.RECORD)

    @patch("mindspore.log.error")
    def test_should_profiler_step_when_no_schedule(self, mock_logger_error):
        """
            Turn off the start_profiler switch and schedule switch, verify the step functions.
        """
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False)

        # The first time the step method is called, the error log should be displayed
        profiler.step()
        mock_logger_error.assert_called_with("With no schedule in the Profiler, step takes no effect!")

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.common.record_function.RecordFunction.start")
    def test_should_profiler_start_stop_when_schedule(self, mock_record_function_start, mock_logger_warning):
        """
            Turn on the schedule switch, turn off start_profile, verify the start---->stop functions.
        """
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False,
                            schedule=Schedule(wait=1, active=1, warmup=1, repeat=1, skip_first=1),
                            data_process=False)

        # The first time the start method is called
        profiler.start()
        self.assertTrue(profiler._has_started)
        self.assertEqual(profiler.current_action, ProfilerAction.NONE)
        mock_record_function_start.assert_called_once()

        # The first time the stop method is called
        profiler.stop()
        self.assertEqual(profiler._has_started, True)
        mock_logger_warning.assert_called_with("The profiler has schedule. Please use step() to collect data.")

    @patch("mindspore.profiler.profiler_action_controller.ProfilerActionController.transit_action")
    @patch("mindspore.profiler.common.record_function.RecordFunction.start")
    @patch("mindspore.profiler.common.record_function.RecordFunction.stop")
    def test_should_profiler_start_step_stop_when_schedule(self, mock_record_function_stop, mock_record_function_start,
                                                           mock_transit_action):
        """
            Turn on the schedule switch, turn off start_profile, verify the start--->step--->stop functions.
        """
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False,
                            schedule=Schedule(wait=1, active=1, warmup=1, repeat=1, skip_first=1),
                            data_process=False)

        # The first time the start method is called
        profiler.start()
        self.assertTrue(profiler._has_started)
        self.assertEqual(profiler.current_action, ProfilerAction.NONE)
        mock_record_function_start.assert_called_once()

        # The first time the step method is called
        profiler.step()
        mock_record_function_stop.assert_called_once()
        self.assertEqual(profiler.current_action, ProfilerAction.NONE)
        mock_transit_action.assert_called_with(ProfilerAction.NONE, ProfilerAction.NONE)
        self.assertEqual(mock_record_function_start.call_count, 2)
        self.assertEqual(profiler._schedule_no_use_step, False)

        # The first time the stop method is called
        profiler.stop()
        self.assertEqual(profiler._has_started, False)
        self.assertEqual(mock_record_function_stop.call_count, 2)
        mock_transit_action.assert_called_with(ProfilerAction.NONE, None)

    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.init")
    @patch("mindspore.profiler.profiler_action_controller.ProfilerActionController.transit_action")
    @patch("mindspore.profiler.common.record_function.RecordFunction.start")
    @patch("mindspore.profiler.common.record_function.RecordFunction.stop")
    def test_should_profiler_for_loop_step_when_schedule_less_than_step(self, mock_record_function_stop,
                                                                        mock_record_function_start, mock_transit_action,
                                                                        mock_prof_interface_init):
        """
            Turn on the schedule switch, turn off start_profile, verify the step--->step--->step... functions.

            details:
                Test case to verify the profile's behavior in a loop with step functions
                when the schedule is less than the step number
        """
        step_num = 30
        expect_actions = [ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.WARM_UP,
                          ProfilerAction.WARM_UP, ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE,
                          ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.WARM_UP, ProfilerAction.WARM_UP,
                          ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE,
                          ProfilerAction.NONE, ProfilerAction.WARM_UP, ProfilerAction.WARM_UP, ProfilerAction.RECORD,
                          ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE, ProfilerAction.NONE,
                          ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.NONE,
                          ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.NONE,
                          ProfilerAction.NONE]
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False,
                            schedule=Schedule(wait=2, active=2, warmup=2, repeat=3, skip_first=2),
                            data_process=False)
        profiler.start()
        action_list = []
        for _ in range(step_num):
            profiler.step()
            action_list.append(profiler.current_action)
        profiler.stop()
        self.assertEqual(action_list, expect_actions)

    @patch("mindspore.profiler.profiler_interface.ProfilerInterface.init")
    @patch("mindspore.profiler.profiler_action_controller.ProfilerActionController.transit_action")
    @patch("mindspore.profiler.common.record_function.RecordFunction.start")
    @patch("mindspore.profiler.common.record_function.RecordFunction.stop")
    def test_should_profiler_for_loop_step_when_schedule_greater_than_step(self, mock_record_function_stop,
                                                                           mock_record_function_start,
                                                                           mock_transit_action,
                                                                           mock_prof_interface_init):
        """
            Turn on the schedule switch, turn off start_profile, verify the step--->step--->step... functions.

            details:
                Test case to verify the profile's behavior in a loop with step functions
                when the schedule is greater than the step number
        """
        step_num = 10
        expect_actions = [ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.WARM_UP,
                          ProfilerAction.WARM_UP, ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE,
                          ProfilerAction.NONE, ProfilerAction.NONE, ProfilerAction.WARM_UP]
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False,
                            schedule=Schedule(wait=2, active=2, warmup=2, repeat=3, skip_first=2),
                            data_process=False)
        profiler.start()
        action_list = []
        for _ in range(step_num):
            profiler.step()
            action_list.append(profiler.current_action)
        profiler.stop()
        self.assertEqual(action_list, expect_actions)

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.common.record_function.RecordFunction.stop")
    @patch("mindspore.profiler.common.record_function.RecordFunction.start")
    @patch("mindspore.profiler.profiler_action_controller.ProfilerActionController.transit_action")
    def test_should_profiler_start_stop_step_when_schedule(self, mock_transit_action, mock_record_function_start,
                                                           mock_record_function_stop, mock_logger_warning):
        """
            Turn on the schedule switch, verify the start--->stop--->step functions.
        """
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False,
                            schedule=Schedule(wait=1, active=1, warmup=1, repeat=1, skip_first=1),
                            data_process=False)

        # The first time the start method is called
        profiler.start()
        self.assertTrue(profiler._has_started)
        self.assertEqual(profiler.current_action, ProfilerAction.NONE)
        mock_record_function_start.assert_called_once()

        # The first time the stop method is called
        profiler.stop()
        self.assertEqual(profiler._has_started, True)
        mock_logger_warning.assert_called_with("The profiler has schedule. Please use step() to collect data.")

        # The first time the step method is called
        profiler.step()
        mock_record_function_stop.assert_called_once()
        self.assertEqual(profiler.current_action, ProfilerAction.NONE)
        mock_transit_action.assert_called_with(ProfilerAction.NONE, ProfilerAction.NONE)
        self.assertEqual(mock_record_function_start.call_count, 2)
        self.assertEqual(profiler._schedule_no_use_step, False)

    @patch("mindspore.log.error")
    def test_should_profiler_no_start_step_when_schedule(self, mock_logger_error):
        """
            Turn on the schedule switch, turn off start_profile, verify the start--->step--->stop functions.
        """
        # Construct the RefactorProfiler instance
        profiler = Profiler(start_profile=False,
                            schedule=Schedule(wait=1, active=1, warmup=1, repeat=1, skip_first=1),
                            data_process=False)

        # The first time the step method is called
        profiler.step()
        mock_logger_error.assert_called_with("Profiler is stopped, step takes no effect!")

    def test_should_profiler_action_when_schedule(self):
        """
            Input the specified step, verify the schedule functionality.
        """
        schedule = Schedule(wait=0, active=2, warmup=0, repeat=2, skip_first=1)
        self.assertEqual(schedule(0), ProfilerAction.NONE)
        self.assertEqual(schedule(1), ProfilerAction.RECORD)

    def test_should_params_correct_when_schedule(self):
        """
            Input the exception schedule, verify the exception log can be printed
            properly, and correct the parameters.
        """
        # Input the parameters of the abnormal type to verify that it can be corrected normally
        schedule = Schedule(wait="1", active=2, warmup="0", repeat=2, skip_first=1)
        self.assertEqual(schedule.to_dict().get('wait'), 0)
        self.assertEqual(schedule.to_dict().get('warmup'), 0)
        # Input the active parameter less than 1 to verify that it can be corrected normally
        schedule_error_active = Schedule(wait=1, active=0, warmup=0, repeat=2, skip_first=1)
        self.assertEqual(schedule_error_active.to_dict().get('active'), 1)

    def test_should_handler_normal_action_correct_when_profiler_action_controller(self):
        """
            Verify whether the ProfActionController can convert actions properly
        """
        profiler_interface = ProfilerInterface()
        prof_action_controller = ProfilerActionController(profiler_interface, None)
        self.assertEqual(prof_action_controller.action_map.get((ProfilerAction.NONE, ProfilerAction.RECORD)),
                         [ProfilerInterface.init, ProfilerInterface.start])
        self.assertEqual(prof_action_controller.action_map.get((ProfilerAction.NONE,
                                                                ProfilerAction.RECORD_AND_SAVE)),
                         [ProfilerInterface.init, ProfilerInterface.start])
