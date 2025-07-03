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
"""Test the AscendMemoryViewer class."""
import unittest
from unittest.mock import patch, MagicMock
import os

from mindspore.profiler.analysis.viewer.ascend_memory_viewer import AscendMemoryViewer, MemoryRecordBean
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.constant import ProfilerActivity


# pylint: disable=protected-access
class TestAscendMemoryViewer(unittest.TestCase):
    @patch.object(ProfilerLogger, 'get_instance')
    def setUp(self, mock_get_instance):
        self.mock_logger = MagicMock()
        self.kwargs = {
            "profile_memory": True,
            "rank_id": 0,
            "ascend_profiler_output_path": "test_output_path",
            "framework_path": "test_framework_path",
            "msprof_profile_output_path": "test_msprof_path",
            "ascend_ms_dir": "test_ascend_ms_dir",
            "activities": [ProfilerActivity.CPU.value]
        }
        mock_get_instance.return_value = self.mock_logger
        self.viewer = AscendMemoryViewer(**self.kwargs)
        self.viewer._logger = ProfilerLogger.get_instance()

    @patch.object(AscendMemoryViewer, '_copy_npu_module_mem_csv')
    @patch.object(AscendMemoryViewer, '_parse_memory_record')
    def test_save_should_success_when_correct(self, mock_parse_memory_record, mock_copy_npu_module_mem_csv):
        self.viewer.save()
        self.viewer._logger.info.assert_any_call("AscendMemoryViewer start")
        mock_copy_npu_module_mem_csv.assert_called_once()
        mock_parse_memory_record.assert_called_once()
        self.viewer._logger.info.assert_any_call("AscendMemoryViewer end")

    def test_save_should_not_end_when_disabled_profile_memory(self):
        self.viewer._enable_profile_memory = False
        self.viewer.save()
        self.viewer._logger.info.assert_called_once_with("AscendMemoryViewer start")

    @patch.object(FileManager, 'get_csv_file_list_by_start_name')
    @patch.object(FileManager, 'combine_csv_file')
    def test_copy_npu_module_mem_csv_should_success_when_correct(self, mock_combine_csv_file, mock_get_csv_file_list):
        mock_get_csv_file_list.return_value = ["file1.csv", "file2.csv"]
        target_file_path = os.path.join(self.kwargs["ascend_profiler_output_path"], "npu_module_mem.csv")
        self.viewer._copy_npu_module_mem_csv()
        mock_get_csv_file_list.assert_called_once_with(self.kwargs["msprof_profile_output_path"], "npu_module_mem")
        mock_combine_csv_file.assert_called_once_with(["file1.csv", "file2.csv"], target_file_path)

    @patch.object(FileManager, 'get_csv_file_list_by_start_name')
    @patch.object(FileManager, 'read_csv_file')
    @patch.object(AscendMemoryViewer, '_parse_ms_memory_record')
    @patch.object(AscendMemoryViewer, '_combine_ge_ms_memory_record')
    @patch.object(FileManager, 'create_csv_file')
    def test_parse_memory_record_should_success_when_correct(
            self, mock_create_csv_file, mock_combine_ge_ms_memory_record,
            mock_parse_ms_memory_record, mock_read_csv_file, mock_get_csv_file_list
        ):
        mock_get_csv_file_list.return_value = ["file1.csv", "file2.csv"]
        mock_read_csv_file.return_value = [["header"], ["data1", "data2"]]
        mock_combine_ge_ms_memory_record.return_value = [["combined_data"]]
        target_file_path = os.path.join(self.kwargs["ascend_profiler_output_path"], "memory_record.csv")
        self.viewer._parse_memory_record()
        mock_get_csv_file_list.assert_called_once_with(self.kwargs["msprof_profile_output_path"], "memory_record")
        mock_read_csv_file.assert_any_call("file1.csv")
        mock_read_csv_file.assert_any_call("file2.csv")
        mock_parse_ms_memory_record.assert_called_once()
        mock_combine_ge_ms_memory_record.assert_called_once()
        mock_create_csv_file.assert_called_once_with(target_file_path, [["combined_data"]],
                                                     self.viewer.TARGET_MEMORY_RECORD_HEADERS)

    @patch.object(FileManager, 'get_csv_file_list_by_start_name')
    @patch.object(FileManager, 'read_csv_file')
    def test_parse_ge_memory_record_should_success_when_correct(self, mock_read_csv_file, mock_get_csv_file_list):
        mock_get_csv_file_list.return_value = ["file1.csv", "file2.csv"]
        mock_read_csv_file.return_value = [["header"], ["data1", "data2"]]
        self.viewer._parse_ge_memory_record()
        mock_get_csv_file_list.assert_called_once_with(self.kwargs["msprof_profile_output_path"], "memory_record")
        mock_read_csv_file.assert_any_call("file1.csv")
        mock_read_csv_file.assert_any_call("file2.csv")
        self.assertEqual(self.viewer._ge_memory_record, [["data1", "data2"], ["data1", "data2"]])

    @patch.object(FileManager, 'read_csv_file')
    def test_parse_ms_memory_record_should_success_when_correct(self, mock_read_csv_file):
        mock_read_csv_file.return_value = [["header"], ["data1", "data2"]]
        self.viewer._parse_ms_memory_record()
        memory_record_file = os.path.join(self.kwargs["framework_path"],
                                          f"cpu_ms_memory_record_{self.kwargs['rank_id']}.txt")
        mock_read_csv_file.assert_called_once_with(memory_record_file)
        self.assertEqual(self.viewer._ms_memory_record, [["data1", "data2"]])

    @patch.object(FileManager, 'read_csv_file')
    def test_parse_ms_memory_should_return_when_record_no_cpu(self, mock_read_csv_file):
        self.viewer._activities = []
        self.viewer._parse_ms_memory_record()
        mock_read_csv_file.assert_not_called()

    @patch.object(FileManager, 'get_csv_file_list_by_start_name')
    @patch.object(FileManager, 'read_csv_file')
    def test_get_app_reserved_memory_should_success_when_correct(self, mock_read_csv_file, mock_get_csv_file_list):
        mock_get_csv_file_list.return_value = ["file1.csv"]
        mock_read_csv_file.return_value = [["id", "APP", "time", "0", "100", "device"]]
        result = self.viewer._get_app_reserved_memory()
        mock_get_csv_file_list.assert_called_once_with(self.kwargs["msprof_profile_output_path"], "npu_mem")
        mock_read_csv_file.assert_called_once_with("file1.csv")
        expected = [
            MemoryRecordBean([
                "APP",
                "device",
                0.0,
                100.0,
                0.0,
                f"NPU:{self.kwargs['rank_id']}"
            ]).row
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
