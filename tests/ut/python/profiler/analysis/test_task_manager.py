import os
import fcntl
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from mindspore.profiler.analysis.task_manager import TaskManager
from mindspore.profiler.analysis.work_flow import WorkFlow
from mindspore.profiler.analysis.parser.base_parser import BaseParser

class TestParser(BaseParser):
    """Test parser implementation."""
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def _parse(self, data):
        # 使用文件锁保证写入安全
        with open(self.output_file, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(f"{self.name}:{str(data)}\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return {"value": data.get("value", 0) + 1} if isinstance(data, dict) else {"value": 1}

class ErrorParser(BaseParser):
    """Test parser implementation."""
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def _parse(self, data):
        # 使用文件锁保证写入安全
        with open(self.output_file, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(f"{self.name}:error\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        raise RuntimeError("Error")

class TestTaskManager(unittest.TestCase):
    """Test cases for TaskManager."""

    def setUp(self):
        """Set up test environment."""
        # 创建 ProfilerLogger mock
        self.logger_patcher = patch('mindspore.profiler.analysis.task_manager.ProfilerLogger')
        self.mock_logger = self.logger_patcher.start()
        # 配置 mock logger 的 get_instance 方法返回值
        self.mock_logger_instance = MagicMock()
        self.mock_logger.get_instance.return_value = self.mock_logger_instance

        # 确保所有日志方法都被mock
        self.mock_logger_instance.info = MagicMock()
        self.mock_logger_instance.error = MagicMock()
        self.mock_logger_instance.warning = MagicMock()
        self.mock_logger_instance.debug = MagicMock()

        # 为 BaseParser 也添加 ProfilerLogger mock
        self.base_logger_patcher = patch('mindspore.profiler.analysis.parser.base_parser.ProfilerLogger')
        self.mock_base_logger = self.base_logger_patcher.start()
        self.mock_base_logger_instance = MagicMock()
        self.mock_base_logger.get_instance.return_value = self.mock_base_logger_instance

        # 确保 BaseParser 的所有日志方法都被mock
        self.mock_base_logger_instance.info = MagicMock()
        self.mock_base_logger_instance.error = MagicMock()
        self.mock_base_logger_instance.warning = MagicMock()
        self.mock_base_logger_instance.debug = MagicMock()

        self.task_manager = TaskManager()
        # 创建临时文件用于验证Parser执行
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.output_file = self.temp_file.name
        TestParser.output_file = self.output_file
        ErrorParser.output_file = self.output_file

    def tearDown(self):
        """Clean up test environment."""
        # 停止所有 mock
        self.logger_patcher.stop()
        self.base_logger_patcher.stop()
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_create_flow_should_success_when_input_valid_parsers(self):
        """Test create_flow with valid parsers."""
        parser1 = TestParser("parser1")
        parser2 = TestParser("parser2")

        # 测试创建work_flow
        self.task_manager.create_flow(parser1, parser2, flow_name="test_flow")

        # 验证work_flow是否创建成功
        self.assertIn("test_flow", self.task_manager.workflows)
        workflow = self.task_manager.workflows["test_flow"]
        self.assertIsInstance(workflow, WorkFlow)
        self.assertEqual(workflow.head_parser, parser1)
        self.assertEqual(parser1.next_parser, parser2)

    def test_create_flow_should_fail_when_input_invalid_parser(self):
        """Test create_flow with invalid parser."""
        parser1 = TestParser("parser1")
        invalid_parser = "not a parser"

        # 验证无效Parser会引发ValueError
        with self.assertRaises(ValueError):
            self.task_manager.create_flow(parser1, invalid_parser, flow_name="test_flow")

    def test_create_flow_should_fail_when_no_parsers(self):
        """Test create_flow with no parsers."""
        # 验证没有提供Parser时的行为
        self.task_manager.create_flow(flow_name="test_flow")
        self.assertNotIn("test_flow", self.task_manager.workflows)

    def test_run_should_execute_parsers_when_flow_is_valid(self):
        """Test run method with valid flow."""
        parser1 = TestParser("parser1")
        parser2 = TestParser("parser2")

        # 创建并运行work_flow
        self.task_manager.create_flow(parser1, parser2, flow_name="test_flow")
        self.task_manager.run()

        # 验证Parser是否按顺序执行
        with open(self.output_file, 'r') as f:
            output = f.read().strip().split('\n')

        self.assertEqual(len(output), 2)
        self.assertEqual(output[0], "parser1:{}")
        self.assertEqual(output[1], "parser2:{'value': 1}")

    def test_run_should_handle_error_when_parser_fails(self):
        """Test run method with failing parser."""
        parser1 = TestParser("parser1")
        parser2 = ErrorParser("parser2")
        parser3 = TestParser("parser3")

        # 创建并运行work_flow
        self.task_manager.create_flow(parser1, parser2, parser3, flow_name="test_flow")
        self.task_manager.run()

        # 验证执行顺序和错误处理
        with open(self.output_file, 'r') as f:
            output = f.read().strip().split('\n')

        self.assertEqual(len(output), 3)
        self.assertEqual(output[0], "parser1:{}")
        self.assertEqual(output[1], "parser2:error")
        self.assertEqual(output[2], "parser3:{'value': 1}")

    def test_run_should_execute_success_when_multiple_flows(self):
        """Test run method with multiple flows."""
        parser1 = TestParser("parser1")
        parser2 = TestParser("parser2")
        parser3 = TestParser("parser3")

        self.task_manager.create_flow(parser1, parser2, flow_name="test_flow1")
        self.task_manager.create_flow(parser3, flow_name="test_flow2")

        self.task_manager.run()

        with open(self.output_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                output = f.read().strip().split('\n')
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self.assertEqual(len(output), 3)
        expected_outputs = {"parser1:{}", "parser2:{'value': 1}", "parser3:{}"}
        self.assertEqual(set(output), expected_outputs)


if __name__ == '__main__':
    unittest.main()
