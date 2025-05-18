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
"""Test dynamic profiler config context."""
from unittest.mock import patch
from mindspore.profiler.dynamic_profile.dynamic_profiler_config_context import DynamicProfilerConfigContext
from mindspore.profiler.common.constant import ProfilerActivity, ExportType, AicoreMetrics, ProfilerLevel


class TestDynamicProfilerConfigContext:
    """Test cases for DynamicProfilerConfigContext class."""

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_parse_all_fields_correctly_when_given_valid_json_data(self, mock_is_dyno):
        """Test initialization with normal valid input data."""
        mock_is_dyno.return_value = True
        json_data = {
            "start_step": "5",
            "iterations": "10",
            "profiler_level": "Level1",
            "activities": "CPU, NPU",
            "export_type": "Text",
            "profile_memory": "true",
            "msprof_tx": "false",
            "with_stack": "True",
            "data_simplification": "False",
            "l2_cache": "true",
            "analyse": "False",
        }
        config = DynamicProfilerConfigContext(json_data).to_dict()

        assert config["start_step"] == 5
        assert config["stop_step"] == 14
        assert config["profiler_level"] == "Level1"
        assert config["activities"] == ["CPU", "NPU"]
        assert config["export_type"] == ["text"]
        assert config["profile_memory"] is True
        assert config["mstx"] is False
        assert config["with_stack"] is True
        assert config["data_simplification"] is False
        assert config["l2_cache"] is True
        assert config["analyse"] is False
        assert config["analyse_mode"] == -1
        assert config["is_valid"] is False
        assert config["aic_metrics"] == "AiCoreNone"
        assert config["parallel_strategy"] is False

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_use_direct_values_when_in_non_dyno_mode(self, mock_is_dyno):
        """Test initialization in non-dynamic profiler mode."""
        mock_is_dyno.return_value = False
        json_data = {
            "start_step": 10,
            "stop_step": 20,
            "activities": ["CPU"],
            "export_type": ["db"],
            "profile_memory": True,
            "mstx": False,
            "parallel_strategy": True,
            "with_stack": False,
            "data_simplification": True,
            "analyse": True,
            "is_valid": True
        }
        config = DynamicProfilerConfigContext(json_data).to_dict()

        assert config["start_step"] == 10
        assert config["stop_step"] == 20
        assert config["profiler_level"] == "Level0"
        assert config["activities"] == ["CPU"]
        assert config["export_type"] == ["db"]
        assert config["profile_memory"] is True
        assert config["mstx"] is False
        assert config["with_stack"] is False
        assert config["data_simplification"] is True
        assert config["l2_cache"] is False
        assert config["analyse"] is True
        assert config["analyse_mode"] == -1
        assert config["is_valid"] is True
        assert config["aic_metrics"] == "AiCoreNone"
        assert config["parallel_strategy"] is True

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_use_default_value_and_log_warning_when_start_step_is_invalid_string(self, mock_is_dyno,
                                                                                        mock_logger_warning):
        """Test handling of invalid string input for start_step."""
        mock_is_dyno.return_value = True
        json_data = {"start_step": "abc"}
        config = DynamicProfilerConfigContext(json_data)
        assert config.start_step == 0
        mock_logger_warning.assert_called_with("dyno config 'start-step' should be an integer, "
                                               "will be reset to default value: '0'.")

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_convert_to_int_when_non_special_param_has_type_error(self, mock_is_dyno):
        """Test type conversion error handling for non-special parameters."""
        mock_is_dyno.return_value = True
        json_data = {"start_step": "abc", "iterations": "xyz"}
        config = DynamicProfilerConfigContext(json_data)
        assert isinstance(config.start_step, int)

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_use_default_activities_when_given_invalid_activity(self, mock_is_dyno, mock_logger_warning):
        """Test conversion of invalid activities input."""
        mock_is_dyno.return_value = False
        json_data = {"activities": ["ABC"]}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["activities"] == [ProfilerActivity.CPU, ProfilerActivity.NPU]
        mock_logger_warning.assert_called_with("'ABC' is not a valid ProfilerActivity member. "
                                               "will be reset to default: '[<ProfilerActivity.CPU: 'CPU'>, "
                                               "<ProfilerActivity.NPU: 'NPU'>]'.")

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_use_default_export_type_when_given_invalid_export_type(self, mock_is_dyno, mock_logger_warning):
        """Test conversion of invalid export type input."""
        mock_is_dyno.return_value = False
        json_data = {"export_type": ["csv"]}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["export_type"] == [ExportType.Text]
        mock_logger_warning.assert_called_with("'csv' is not a valid ExportType member. "
                                               "will be reset to default: '[<ExportType.Text: 'text'>]'.")

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_maintain_same_value_when_converting_between_json_and_bytes(self, mock_is_dyno):
        """Test serialization to bytes and deserialization back to JSON."""
        mock_is_dyno.return_value = True
        json_data = {"start_step": "5"}
        config = DynamicProfilerConfigContext(json_data)
        bytes_data = config.to_bytes()
        recovered = DynamicProfilerConfigContext.bytes_to_json(bytes_data)
        assert recovered["start_step"] == 5

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_convert_to_level1_when_profiler_level_is_1(self, mock_is_dyno):
        """Test integer input conversion for profiler level."""
        mock_is_dyno.return_value = True
        json_data = {"profiler_level": 1}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["profiler_level"] == ProfilerLevel.Level1

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_convert_to_cpu_when_activities_is_1(self, mock_is_dyno):
        """Test integer input conversion for activities."""
        mock_is_dyno.return_value = False
        json_data = {"activities": 1}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["activities"] == [ProfilerActivity.CPU]

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_convert_to_db_when_export_type_is_1(self, mock_is_dyno):
        """Test integer input conversion for export type."""
        mock_is_dyno.return_value = False
        json_data = {"export_type": 1}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["export_type"] == [ExportType.Db]

    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_convert_to_pipeutilization_when_aic_metrics_is_0(self, mock_is_dyno):
        """Test integer input conversion for AIC metrics."""
        mock_is_dyno.return_value = True
        json_data = {"aic_metrics": 0}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["aic_metrics"] == AicoreMetrics.PipeUtilization

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_use_default_level_and_log_warning_when_profiler_level_is_invalid(self, mock_is_dyno,
                                                                                     mock_logger_warning):
        """Test handling of invalid profiler level input."""
        mock_is_dyno.return_value = True
        json_data = {"profiler_level": "InvalidLevel"}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["profiler_level"] == ProfilerLevel.Level0
        mock_logger_warning.assert_called_with("'InvalidLevel' is not a valid profiler_level, "
                                               "will be reset to will be reset to default: 'Level0'.")

    @patch("mindspore.log.warning")
    @patch("mindspore.profiler.dynamic_profile.dynamic_profiler_utils.DynamicProfilerUtils.is_dyno_mode")
    def test_should_use_default_metric_and_log_warning_when_aic_metrics_is_invalid(self, mock_is_dyno,
                                                                                   mock_logger_warning):
        """Test handling of invalid AIC metrics input."""
        mock_is_dyno.return_value = True
        json_data = {"aic_metrics": "InvalidMetric"}
        config = DynamicProfilerConfigContext(json_data)
        assert config.args["aic_metrics"] == AicoreMetrics.AiCoreNone
        mock_logger_warning.assert_called_with("'InvalidMetric' is not a valid aic_metrics, "
                                               "will be reset to will be reset to default: 'AiCoreNone'.")
