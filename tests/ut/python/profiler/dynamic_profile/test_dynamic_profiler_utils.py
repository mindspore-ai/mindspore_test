# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Test dynamic profiler utils."""
import os
import unittest
from unittest import mock
from unittest.mock import patch
from mindspore.profiler.common.constant import DynoMode
from mindspore.profiler.dynamic_profile.dynamic_profiler_utils import DynamicProfilerUtils


class TestDynamicProfilerUtils(unittest.TestCase):
    """Unit tests for DynamicProfilerUtils class."""

    def test_should_return_true_when_dyno_mode_is_enabled(self):
        """Test that is_dyno_mode returns True when DYNO_DAEMON is set to 1."""
        with mock.patch.dict(os.environ, {DynoMode.DYNO_DAEMON: "1"}):
            self.assertTrue(DynamicProfilerUtils.is_dyno_mode())

    def test_should_return_false_when_dyno_mode_is_disabled(self):
        """Test that is_dyno_mode returns False when DYNO_DAEMON is set to 0."""
        with mock.patch.dict(os.environ, {DynoMode.DYNO_DAEMON: "0"}):
            self.assertFalse(DynamicProfilerUtils.is_dyno_mode())

    @patch("mindspore.log.error")
    def test_should_return_false_and_log_error_when_dyno_mode_has_invalid_value(self, mock_logger_error):
        """Test that is_dyno_mode returns False and logs error when DYNO_DAEMON has invalid value."""
        with mock.patch.dict(os.environ, {DynoMode.DYNO_DAEMON: "abc"}):
            self.assertFalse(DynamicProfilerUtils.is_dyno_mode())
            mock_logger_error.assert_called_with("Environment variable 'MSMONITOR_USE_DAEMON' "
                                                 "value not valid, will be set to 0 !")

    @mock.patch("mindspore.communication.get_rank", side_effect=RuntimeError("Mock error"))
    def test_should_return_zero_when_get_rank_fails_and_rank_id_missing(self, mock_get_rank):
        """Test that get_real_rank returns 0 when get_rank fails and RANK_ID is missing."""
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(DynamicProfilerUtils.get_real_rank(), 0)

    def test_should_parse_correct_dict_when_dyno_str_has_normal_format(self):
        """Test that dyno_str_to_dict correctly parses well-formatted input string."""
        input_str = "PROFILE_START_STEP=10\nPROFILE_ACTIVITIES=CPU,NPU\nPROFILE_L2_CACHE=false"
        expected = {
            "start_step": "10",
            "activities": "CPU,NPU",
            "l2_cache": "false"
        }
        result = DynamicProfilerUtils.dyno_str_to_dict(input_str)
        self.assertEqual(result, expected)

    def test_should_ignore_empty_lines_when_parsing_dyno_str(self):
        """Test that dyno_str_to_dict ignores empty lines in input string."""
        input_str = "\nPROFILE_START_STEP=10\n\nPROFILE_L2_CACHE=false"
        expected = {
            "start_step": "10",
            "l2_cache": "false"
        }
        result = DynamicProfilerUtils.dyno_str_to_dict(input_str)
        self.assertEqual(result, expected)

    def test_should_skip_malformed_lines_when_parsing_dyno_str(self):
        """Test that dyno_str_to_dict skips malformed lines in input string."""
        input_str = "PROFILE_START_STEP\nPROFILE_L2_CACHE=false"
        expected = {
            "l2_cache": "false"
        }
        result = DynamicProfilerUtils.dyno_str_to_dict(input_str)
        self.assertEqual(result, expected)

    def test_should_handle_mixed_case_keys_when_parsing_dyno_str(self):
        """Test that dyno_str_to_dict handles mixed case keys in input string."""
        input_str = "PROFILE_start_step=10\nactivities=NPU\nPROFILE_l2_cache=false"
        expected = {
            "start_step": "10",
            "activities": "NPU",
            "l2_cache": "false"
        }
        result = DynamicProfilerUtils.dyno_str_to_dict(input_str)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
