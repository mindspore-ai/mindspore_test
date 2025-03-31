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
"""Test the FrameworkParser class."""
import unittest
from unittest.mock import patch

from mindspore.profiler.analysis.parser.ms_framework_parser import FrameworkParser
from mindspore.profiler.common.constant import ProfilerActivity, DeviceTarget
from mindspore.profiler.common.log import ProfilerLogger


# pylint: disable=protected-access
class TestFrameworkParser(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "rank_id": 0,
            "activities": [ProfilerActivity.CPU.value, ProfilerActivity.NPU.value],
            "step_list": [1, 2, 3],
            "framework_path": "test_framework_path",
            "device_target": DeviceTarget.NPU.value,
            "ascend_ms_dir": "test_ascend_ms_dir"
        }

    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    @patch.object(FrameworkParser, '_parse_op_range_data')
    @patch.object(FrameworkParser, '_parse_cpu_op_data')
    def test_parse_should_success_when_correct(
            self, mock_parse_cpu_op_data, mock_parse_op_range_data, mock_get_instance, mock_init
        ):
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        mock_init.return_value = None
        mock_parse_op_range_data.return_value = [{"op": "test_op"}]
        mock_parse_cpu_op_data.return_value = ["line1", "line2"]
        parser = FrameworkParser(**self.kwargs)
        data = {}
        result = parser._parse(data)
        mock_parse_op_range_data.assert_called_once()
        mock_parse_cpu_op_data.assert_called_once()
        self.assertEqual(result["mindspore_op_list"], [{"op": "test_op"}])
        self.assertEqual(result["cpu_op_lines"], ["line1", "line2"])

    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    @patch.object(FrameworkParser, '_parse_op_range_data')
    @patch.object(FrameworkParser, '_parse_cpu_op_data')
    def test_parse_should_return_when_activity_only_npu(
            self, mock_parse_cpu_op_data, mock_parse_op_range_data, mock_get_instance, mock_init
        ):
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        mock_init.return_value = None
        new_kwargs = self.kwargs.copy()
        new_kwargs["activities"] = [ProfilerActivity.NPU.value]
        parser = FrameworkParser(**new_kwargs)
        data = {}
        result = parser._parse(data)
        mock_parse_op_range_data.assert_not_called()
        mock_parse_cpu_op_data.assert_not_called()
        self.assertEqual(result, data)

    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    @patch.object(FrameworkParser, '_parse_op_range_data')
    @patch.object(FrameworkParser, '_parse_cpu_op_data')
    def test_parse_should_return_when_device_target_cpu(
            self, mock_parse_cpu_op_data, mock_parse_op_range_data, mock_get_instance, mock_init
        ):
        mock_parse_op_range_data.return_value = []
        mock_parse_cpu_op_data.return_value = ["line1", "line2"]
        new_kwargs = self.kwargs.copy()
        new_kwargs["device_target"] = DeviceTarget.CPU.value
        parser = FrameworkParser(**new_kwargs)
        data = {}
        result = parser._parse(data)
        mock_parse_op_range_data.assert_called_once()
        mock_parse_cpu_op_data.assert_called_once()
        self.assertEqual(result["mindspore_op_list"], [])
        self.assertEqual(result["cpu_op_lines"], ["line1", "line2"])

    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    @patch.object(FrameworkParser, '_filter_op_range_list')
    @patch("mindspore.profiler.common.file_manager.FileManager.read_file_content")
    @patch("mindspore.profiler.common.tlv_decoder.TLVDecoder.decode")
    def test_parse_op_range_data_should_success_when_correct(
            self, mock_decode, mock_read_file_content,
            mock_filter_op_range_list, mock_get_instance, mock_init
        ):
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        mock_init.return_value = None
        mock_read_file_content.return_value = b"test_bytes"
        mock_decode.return_value = [{"op": "test_op"}]
        mock_filter_op_range_list.return_value = [{"op": "test_op"}]
        parser = FrameworkParser(**self.kwargs)
        result = None
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            result = parser._parse_op_range_data()

        mock_read_file_content.assert_called_once()
        mock_decode.assert_called_once()
        mock_filter_op_range_list.assert_called_once_with([{"op": "test_op"}])
        self.assertEqual(result, [{"op": "test_op"}])

    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    def test_filter_op_range_list_should_success_when_correct(self, mock_get_instance, mock_init):
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        mock_init.return_value = None
        parser = FrameworkParser(**self.kwargs)
        op_range_list = [
            {"fix_size_data": [0.0, 1.0, 1]},
            {"fix_size_data": [1.1, 2.2, 2]},
            {"fix_size_data": [3.3, 4.4, 4]}
        ]
        result = parser._filter_op_range_list(op_range_list)
        expected = [
            {"fix_size_data": [0.0, 1.0, 1]},
            {"fix_size_data": [1.1, 2.2, 2]}
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
