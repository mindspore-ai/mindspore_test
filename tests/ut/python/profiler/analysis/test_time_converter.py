# pylint: disable=protected-access
import unittest
from decimal import Decimal
from mindspore.profiler.analysis.time_converter import TimeConverter


class TestTimeConverter(unittest.TestCase):
    def setUp(self):
        freq = 100.0
        cntvct = 2000000000
        localtime_diff = 1000000000
        TimeConverter.init_parameters(freq, cntvct, localtime_diff)

    def test_convert_syscnt_to_timestamp_us_should_calculate_correctly_when_syscnt_is_valid(self):
        result = TimeConverter.convert_syscnt_to_timestamp_us(30000000000)
        self.assertIsInstance(result, Decimal)
        self.assertEqual(result, Decimal("281000000.000"))

        result = TimeConverter.convert_syscnt_to_timestamp_us(20000000000)
        self.assertEqual(result, Decimal("181000000.000"))

    def test_convert_syscnt_should_raise_error_when_without_init(self):
        TimeConverter._is_loaded = False
        with self.assertRaises(RuntimeError):
            TimeConverter.convert_syscnt_to_timestamp_us(30000000000)

    def tearDown(self):
        TimeConverter._is_loaded = False

if __name__ == "__main__":
    unittest.main()
