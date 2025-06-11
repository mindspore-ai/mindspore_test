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
"""Test the MsOperatorDetailsViewer class."""
import os
import unittest
from unittest.mock import patch, MagicMock

from mindspore.profiler.analysis.viewer.ms_operator_details_viewer import MsOperatorDetailsViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.log import ProfilerLogger


# pylint: disable=protected-access
class TestMsOperatorDetailsViewer(unittest.TestCase):
    """Unit tests for MsOperatorDetailsViewer class."""

    def setUp(self):
        """Test environment setup."""
        self.mock_logger = MagicMock()
        ProfilerLogger.get_instance = MagicMock(return_value=self.mock_logger)

        self.profiler_output = "profiler_output"
        self.ascend_ms_dir = "ascend_ms_dir"
        self.framework_path = "framework_path"
        self._COL_NAMES = ['Name', 'Input Shapes']

        self.viewer = MsOperatorDetailsViewer(
            ascend_profiler_output_path=self.profiler_output,
            ascend_ms_dir=self.ascend_ms_dir,
            framework_path=self.framework_path
        )

    @patch('os.path.isfile')
    @patch.object(FileManager, 'read_file_content')
    @patch('mindspore.profiler.common.tlv_decoder.TLVDecoder.decode')
    def test_read_fwk_binary_file_should_success_when_correct(self, mock_decode, mock_read, mock_isfile):
        """Test reading framework binary file successfully."""
        mock_isfile.return_value = True
        mock_read.return_value = b'sample_binary_data'
        mock_decode.return_value = [{0: "op1", 1: "[32,256]"}]
        self.viewer._read_fwk_binary_file()

        mock_read.assert_called_with(os.path.join(self.framework_path, "mindspore.record_shapes"), mode="rb")
        self.assertEqual(len(self.viewer._operator_details_events), 1)

    def test_calculate_operator_details_data_should_success_when_correct(self):
        """Test operator data processing."""
        mock_event1 = MagicMock()
        mock_event1.name = "Conv2D"
        mock_event1.input_shapes = "[32,3,224,224]"
        mock_event2 = MagicMock()
        mock_event2.name = "ReLU"
        mock_event2.input_shapes = "[32,64,112,112]"
        self.viewer._operator_details_events = [mock_event1, mock_event2]

        self.viewer._calculate_operator_details_data()

        self.assertEqual(len(self.viewer._operator_details_data), 2)
        self.assertEqual(self.viewer._operator_details_data[0][0], "Conv2D")

    @patch.object(FileManager, 'create_csv_file')
    def test_write_data_should_success_when_correct(self, mock_create_csv):
        """Test CSV file creation."""
        self.viewer._operator_details_data = [
            ["Conv2D", "[32,3,224,224]"],
            ["ReLU", "[32,64,112,112]"]
        ]

        self.viewer._write_data()

        expected_path = os.path.join(self.profiler_output, 'operator_details.csv')
        mock_create_csv.assert_called_with(
            expected_path,
            self.viewer._operator_details_data,
            self._COL_NAMES
        )

if __name__ == '__main__':
    unittest.main()
