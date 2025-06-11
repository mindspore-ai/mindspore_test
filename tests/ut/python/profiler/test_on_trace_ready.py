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
from mindspore.profiler.profiler import tensorboard_trace_handler
from mindspore.profiler.profiler_interface import ProfilerInterface
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager


# pylint: disable=protected-access
class TestProfiler(unittest.TestCase):
    """Test the on_trace_ready function."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch("mindspore.log.warning")
    def test_should_tensorboard_trace_handler_correct_when_execute(self, mock_logger_warning):
        """
            Verify whether the tensorboard_trace_handler function executes normally
        """
        Profiler(start_profile=False)
        ProfilerPathManager()
        ProfilerPathManager().init = unittest.mock.Mock()
        tensorboard_trace_handler(analyse_flag="True")
        mock_logger_warning.assert_called_with("analyse_flag is not bool, set by default.")
        ProfilerPathManager().init.assert_called_once()

        ProfilerPathManager().init = unittest.mock.Mock()
        tensorboard_trace_handler(async_mode="True")
        mock_logger_warning.assert_called_with("async_mode is not bool, set by default.")
        ProfilerPathManager().init.assert_called_once()

    @patch("mindspore.profiler.profiler.tensorboard_trace_handler")
    def test_should_tensorboard_trace_handler_correct_when_called(self, mock_tensorboard_trace_handler):
        """
            Verify the behavior when tensorboard_trace_handler is called
        """
        # Correct registration of trace handler as callback
        profiler = Profiler(start_profile=False, on_trace_ready=mock_tensorboard_trace_handler)
        prof_action_controller = profiler.action_controller
        self.assertEqual(
            prof_action_controller.action_map.get((ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE)),
            [
                ProfilerInterface.stop,
                ProfilerInterface.finalize,
                prof_action_controller._trace_ready,
                ProfilerInterface.clear
            ])
        self.assertEqual(prof_action_controller.on_trace_ready, mock_tensorboard_trace_handler)
        prof_action_controller._trace_ready()
        mock_tensorboard_trace_handler.assert_called_once()

        # Error handling when the callback is not callable
        test_cases = [
            (None, "None value"),
            (1, "numeric value"),
            ("", "empty string"),
        ]
        for callback, description in test_cases:
            with self.subTest(msg="Testing " + description):
                mock_tensorboard_trace_handler.reset_mock()
                profiler = Profiler(start_profile=False, on_trace_ready=callback)
                prof_action_controller = profiler.action_controller
                prof_action_controller._trace_ready()
                mock_tensorboard_trace_handler.assert_not_called()
