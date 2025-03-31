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
"""Test the AscendMsprofParser class."""
import unittest
from unittest.mock import patch
import glob
import numpy as np

from mindspore.profiler.analysis.parser.ascend_cann_parser import AscendMsprofParser
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.constant import ProfilerLevel, ExportType
from mindspore.profiler.common.log import ProfilerLogger


# pylint: disable=protected-access
class TestAscendMsprofParser(unittest.TestCase):
    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    def setUp(self, mock_get_instance, mock_init):
        self.kwargs = {
            "msprof_profile_output_path": "test_output_path",
            "msprof_profile_host_path": "test_host_path",
            "msprof_profile_device_path": "test_device_path",
            "ascend_ms_dir": "test_ascend_ms_dir",
            "profiler_level": ProfilerLevel.Level1.value,
            "export_type": [ExportType.Text.value]
        }
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        mock_init.return_value = None
        self.parser = AscendMsprofParser(**self.kwargs)

    @patch.object(glob, 'glob')
    @patch.object(FileManager, 'read_csv_file_as_numpy')
    def test_parse_op_summary_should_success_when_correct(self, mock_read_csv, mock_glob):
        mock_glob.return_value = ["test_op_summary.csv"]
        mock_read_csv.return_value = (np.array([[1, 2]]), ["header1", "header2"])
        self.parser._parse_op_summary()
        self.assertEqual(self.parser.op_summary.tolist(), [[1, 2]])
        self.assertEqual(self.parser.op_summary_headers, ["header1", "header2"])

    def test_parse_op_summary_should_return_when_exporttype_db(self):
        new_parser = self.parser
        new_parser.op_summary = None
        new_parser.op_summary_headers = None
        new_parser._export_type = [ExportType.Db.value]
        new_parser._parse_op_summary()
        self.assertEqual(new_parser.op_summary, None)
        self.assertEqual(new_parser.op_summary_headers, None)

    def test_parse_op_summary_should_return_when_levelnone(self):
        new_parser = self.parser
        new_parser.op_summary = None
        new_parser.op_summary_headers = None
        new_parser.profiler_level = ProfilerLevel.LevelNone.value
        new_parser._parse_op_summary()
        self.assertEqual(new_parser.op_summary, None)
        self.assertEqual(new_parser.op_summary_headers, None)

    @patch.object(glob, 'glob')
    @patch.object(FileManager, 'read_json_file')
    def test_parse_msprof_timeline_should_success_when_correct(self, mock_read_json, mock_glob):
        mock_glob.return_value = ["test_msprof.json"]
        mock_read_json.return_value = [{
            "name": "task1",
            "ph": "X",
            "ts": 1000,
            "dur": 100,
            "pid": 1234
        }]
        self.parser.msprof_timeline = []
        self.parser._parse_msprof_timeline()
        self.assertEqual(len(self.parser.msprof_timeline), 2)
        self.assertIn("name", self.parser.msprof_timeline[0])
        self.assertEqual(self.parser.msprof_timeline[0]["dur"], 100)

    def test_parse_msprof_timeline_should_return_when_exporttype_db(self):
        new_parser = self.parser
        new_parser.msprof_timeline = []
        new_parser._export_type = [ExportType.Db.value]
        new_parser._parse_msprof_timeline()
        self.assertEqual(new_parser.msprof_timeline, [])

    @patch.object(PathManager, 'get_directory_size')
    @patch('mindspore.log.warning')
    def test_check_msprof_data_size_should_warning_when_bigsize(self, mock_logger, mock_get_size):
        mock_get_size.side_effect = [1024, 1024]
        self.parser._check_msprof_data_size()
        mock_logger.assert_called_once()


if __name__ == "__main__":
    unittest.main()
