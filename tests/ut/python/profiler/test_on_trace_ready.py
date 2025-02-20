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
"""Test the on_trace_ready function."""
import unittest
from unittest.mock import patch
import unittest.mock

from mindspore.profiler.profiler import Profiler
from mindspore.profiler.schedule import ProfilerAction
from mindspore.profiler.profiler import tensor_board_trace_handler
from mindspore.profiler.profiler_interface import ProfilerInterface


# pylint: disable=protected-access
class TestProfiler(unittest.TestCase):
    """Test the on_trace_ready function."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch("mindspore.log.error")
    @patch("mindspore.profiler.platform.npu_profiler.NPUProfilerAnalysis.online_analyse")
    def test_should_tensor_board_trace_handler_correct_when_execute(self,
                                                                    mock_npu_profiler_analysis_online_analyse,
                                                                    mock_logger_error):
        """
            Verify whether the tensor_board_trace_handler function executes normally
        """
        # Normal execution path with data simplification
        Profiler(start_profile=False)
        tensor_board_trace_handler()
        mock_npu_profiler_analysis_online_analyse.assert_called_once()
        mock_logger_error.assert_not_called()

        # Normal execution path without data simplification
        mock_npu_profiler_analysis_online_analyse.reset_mock()
        Profiler(start_profile=False, data_simplification=False)
        tensor_board_trace_handler()
        mock_npu_profiler_analysis_online_analyse.assert_called_once()
        mock_logger_error.assert_not_called()

        # Error handling when an exception occurs
        mock_npu_profiler_analysis_online_analyse.reset_mock()
        mock_npu_profiler_analysis_online_analyse.side_effect = Exception("Test error")
        tensor_board_trace_handler()
        mock_npu_profiler_analysis_online_analyse.assert_called_once()
        mock_logger_error.assert_called_once_with(
            "Call tensorboard_trace_handler failed. Exception: %s", "Test error")

    @patch("mindspore.profiler.profiler.tensor_board_trace_handler")
    def test_should_tensor_board_trace_handler_correct_when_called(self, mock_tensor_board_trace_handler):
        """
            Verify the behavior when tensor_board_trace_handler is called
        """
        # Correct registration of trace handler as callback
        profiler = Profiler(start_profile=False, on_trace_ready=mock_tensor_board_trace_handler)
        prof_action_controller = profiler.action_controller
        self.assertEqual(
            prof_action_controller.handle_normal_action(ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE),
            [
                ProfilerInterface.stop,
                ProfilerInterface.finalize,
                prof_action_controller._trace_ready,
                ProfilerInterface.clear
            ])
        self.assertEqual(prof_action_controller.on_trace_ready, mock_tensor_board_trace_handler)
        prof_action_controller._trace_ready()
        mock_tensor_board_trace_handler.assert_called_once()

        # Error handling when the callback is not callable
        test_cases = [
            (None, "None value"),
            (1, "numeric value"),
            ("", "empty string"),
        ]
        for callback, description in test_cases:
            with self.subTest(msg="Testing " + description):
                mock_tensor_board_trace_handler.reset_mock()
                profiler = Profiler(start_profile=False, on_trace_ready=callback)
                prof_action_controller = profiler.action_controller
                self.assertIsNone(prof_action_controller.on_trace_ready)
                prof_action_controller._trace_ready()
                mock_tensor_board_trace_handler.assert_not_called()
