# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Unit tests for FileManager."""
import os
import shutil
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from mindspore.profiler.common.file_manager import FileManager


class TestFileManager(unittest.TestCase):
    """Test FileManager class."""

    def setUp(self):
        """Create temporary directory and files for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix='test_file_manager_')
        self.test_dir = os.path.join(self.temp_dir, 'test_dir')
        os.makedirs(self.test_dir)

        # 创建测试文件和数据
        self.json_file = os.path.join(self.test_dir, "test.json")
        self.csv_file = os.path.join(self.test_dir, "test.csv")
        self.txt_file = os.path.join(self.test_dir, "test.txt")
        self.test_json_data = [{"name": "test", "value": 1}]

        # 写入测试数据
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.test_json_data, f)

        with open(self.csv_file, "w", encoding="utf-8") as f:
            f.write("name,value\ntest,1\n")

        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write("test line 1\ntest line 2")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_read_file_content_should_success_when_file_exists(self):
        """Test read_file_content with existing file."""
        content = FileManager.read_file_content(self.txt_file)
        self.assertEqual(content, "test line 1\ntest line 2")

    def test_read_file_content_should_raise_exception_when_file_not_exists(self):
        """Test read_file_content with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            FileManager.read_file_content(os.path.join(self.test_dir, "nonexistent.txt"))

    def test_read_json_file_should_success_when_file_valid(self):
        """Test read_json_file with valid JSON file."""
        data = FileManager.read_json_file(self.json_file)
        self.assertEqual(data, self.test_json_data)

    def test_read_json_file_should_raise_exception_when_content_invalid(self):
        """Test read_json_file with invalid JSON content."""
        invalid_json = os.path.join(self.test_dir, "invalid.json")
        with open(invalid_json, "w", encoding="utf-8") as f:
            f.write("invalid json")
        with self.assertRaises(RuntimeError) as cm:
            FileManager.read_json_file(invalid_json)
        self.assertIn("Failed read json file", str(cm.exception))

    def test_create_json_file_should_success_when_data_valid(self):
        """Test create_json_file with valid data."""
        output_file = os.path.join(self.test_dir, "output.json")
        FileManager.create_json_file(output_file, self.test_json_data)

        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, self.test_json_data)

    def test_create_json_file_should_not_create_when_data_empty(self):
        """Test create_json_file with empty data."""
        output_file = os.path.join(self.test_dir, "empty.json")
        FileManager.create_json_file(output_file, [])
        self.assertFalse(os.path.exists(output_file))

    def test_read_csv_file_should_success_when_file_valid(self):
        """Test read_csv_file with valid CSV file."""
        data = FileManager.read_csv_file(self.csv_file)
        self.assertEqual(data, [["name", "value"], ["test", "1"]])

    def test_read_csv_file_should_return_empty_when_file_empty(self):
        """Test read_csv_file with empty CSV file."""
        empty_csv = os.path.join(self.test_dir, "empty.csv")
        with open(empty_csv, "w", encoding="utf-8"):
            pass
        data = FileManager.read_csv_file(empty_csv)
        self.assertEqual(data, [])

    def test_read_csv_file_as_numpy_should_success_when_file_valid(self):
        """Test read_csv_file_as_numpy with valid CSV file."""
        data, headers = FileManager.read_csv_file_as_numpy(self.csv_file)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(headers, ["name", "value"])
        self.assertEqual(data["name"][0], "test")
        self.assertEqual(data["value"][0], "1")

    def test_read_csv_file_as_numpy_should_include_extern_headers_when_provided(self):
        """Test read_csv_file_as_numpy with external headers."""
        extern_headers = ["id"]
        _, headers = FileManager.read_csv_file_as_numpy(self.csv_file, extern_headers)
        self.assertTrue(all(header in headers for header in extern_headers))

    def test_create_csv_file_should_success_when_data_valid(self):
        """Test create_csv_file with valid data."""
        output_file = os.path.join(self.test_dir, "output.csv")
        test_data = [["test", "1"]]
        headers = ["name", "value"]

        FileManager.create_csv_file(output_file, test_data, headers)
        self.assertTrue(os.path.exists(output_file))

        data = FileManager.read_csv_file(output_file)
        self.assertEqual(data, [headers] + test_data)

    def test_create_csv_file_should_not_create_when_data_empty(self):
        """Test create_csv_file with empty data."""
        output_file = os.path.join(self.test_dir, "empty.csv")
        FileManager.create_csv_file(output_file, [])
        self.assertFalse(os.path.exists(output_file))

    def test_combine_csv_file_should_success_when_files_valid(self):
        """Test combine_csv_file with valid CSV files."""
        csv1 = os.path.join(self.test_dir, "test1.csv")
        csv2 = os.path.join(self.test_dir, "test2.csv")
        with open(csv1, "w", encoding="utf-8") as f:
            f.write("name,value\ntest1,1\n")
        with open(csv2, "w", encoding="utf-8") as f:
            f.write("name,value\ntest2,2\n")

        output_file = os.path.join(self.test_dir, "combined.csv")
        header_map = {"name": "new_name", "value": "new_value"}

        FileManager.combine_csv_file([csv1, csv2], output_file, header_map)

        data = FileManager.read_csv_file(output_file)
        self.assertEqual(data[0], ["new_name", "new_value"])
        self.assertEqual(len(data), 3)  # header + 2 rows

    def test_read_txt_file_should_success_when_file_valid(self):
        """Test read_txt_file with valid text file."""
        lines = FileManager.read_txt_file(self.txt_file)
        self.assertEqual(lines, ["test line 1", "test line 2"])

    def test_read_txt_file_should_return_empty_when_file_empty(self):
        """Test read_txt_file with empty text file."""
        empty_txt = os.path.join(self.test_dir, "empty.txt")
        with open(empty_txt, "w", encoding="utf-8"):
            pass
        lines = FileManager.read_txt_file(empty_txt)
        self.assertEqual(lines, [])

    def test_copy_file_should_success_when_source_exists(self):
        """Test copy_file with existing source file."""
        dst_dir = os.path.join(self.test_dir, "dst_dir")
        os.makedirs(dst_dir)
        dst_file = os.path.join(dst_dir, "copied.txt")
        with open(dst_file, 'w', encoding="utf-8") as f:
            f.write("")

        FileManager.copy_file(self.txt_file, dst_file)
        self.assertTrue(os.path.exists(dst_file))
        with open(dst_file, encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "test line 1\ntest line 2")

    def test_copy_file_should_not_copy_when_source_not_exists(self):
        """Test copy_file with non-existent source file."""
        dst_dir = os.path.join(self.test_dir, "dst_dir")
        os.makedirs(dst_dir)
        non_existent = os.path.join(self.test_dir, "non_existent.txt")
        dst_file = os.path.join(dst_dir, "copied.txt")
        with open(dst_file, 'w', encoding="utf-8") as f:
            f.write("")

        with patch('mindspore.log.warning') as mock_warning:
            FileManager.copy_file(non_existent, dst_file)
            mock_warning.assert_called_once_with(
                "The source file does not exist: %s", non_existent)

    @patch('os.stat')
    @patch('os.geteuid')
    def test_check_file_owner_should_return_true_when_file_owner_matches(self, mock_geteuid, mock_stat):
        test_file = "file_owner.json"
        test_path = os.path.join(self.test_dir, test_file)
        mock_geteuid.return_value = 1000
        mock_stat.return_value.st_uid = 0
        self.assertTrue(FileManager.check_file_owner(test_path))
        mock_stat.return_value.st_uid = 1000
        self.assertTrue(FileManager.check_file_owner(test_path))
        mock_stat.return_value.st_uid = 9999
        self.assertFalse(FileManager.check_file_owner(test_path))


if __name__ == "__main__":
    unittest.main()
