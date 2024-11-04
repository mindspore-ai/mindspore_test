# pylint: disable=protected-access
import os
import tempfile
import unittest

from mindspore.profiler.analysis.parser.base_parser import BaseParser


class DummyParser(BaseParser):
    """Test parser implementation of BaseParser."""
    def _parse(self, data):
        return data + 1

# 在类外部定义全局钩子函数
test_result_file = None

def global_test_hook(result):
    """Global hook function for testing."""
    with open(test_result_file, 'w') as f:
        f.write(str(result))

class TestBaseParser(unittest.TestCase):
    """Test cases for BaseParser."""

    def setUp(self):
        """Set up test environment."""
        self.parser1 = DummyParser()
        self.parser2 = DummyParser()
        self.parser3 = DummyParser()

    def test_set_next_should_set_success_when_when_next_parser_type_is_valid(self):
        """Test set_next method."""
        self.parser1.set_next(self.parser2)
        self.assertEqual(self.parser1.next_parser, self.parser2)

    def test_set_next_should_raise_value_error_when_next_parser_type_is_invalid(self):
        """Test set_next method."""
        with self.assertRaises(ValueError):
            self.parser1.set_next("not a parser")

    def test_parse_hook_should_execute_success_when_register_post_hook_is_valid(self):
        """Test parse method and post hook should execute success when register post hook is valid."""
        global test_result_file

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            test_result_file = temp_file.name

            try:
                self.parser1.register_post_hook(global_test_hook)
                self.parser1.parse(1)

                self.assertTrue(os.path.exists(test_result_file))
                with open(test_result_file, 'r') as f:
                    self.assertEqual(f.read(), '2')
            finally:
                if os.path.exists(test_result_file):
                    os.remove(test_result_file)
                test_result_file = None

    def test_register_post_hook_should_register_success_when_hook_is_valid(self):
        """Test register_post_hook method is valid."""
        def valid_hook(result):
            pass

        parser = self.parser1.register_post_hook(valid_hook)
        self.assertEqual(len(self.parser1._post_hooks), 1)
        self.assertEqual(parser, self.parser1)

    def test_register_post_hook_should_raise_value_error_when_hook_is_invalid(self):
        """Test register_post_hook method is invalid."""
        with self.assertRaises(ValueError):
            self.parser1.register_post_hook("not callable")

if __name__ == '__main__':
    unittest.main()
