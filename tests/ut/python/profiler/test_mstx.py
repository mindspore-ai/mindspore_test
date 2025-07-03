import unittest
from unittest.mock import patch, Mock
from mindspore.profiler import mstx
import mindspore
from mindspore import context


class TestMstx(unittest.TestCase):
    """Test cases for mstx class."""

    def setUp(self):
        """Set up test environment."""
        # Create mock Profiler with required methods
        self.mock_profiler = Mock()
        self.mock_profiler.mstx_mark = Mock()
        self.mock_profiler.mstx_range_start = Mock(return_value=1)
        self.mock_profiler.mstx_range_end = Mock()

        # Create mock Stream object
        self.mock_stream = Mock(spec=mindspore.runtime.Stream)
        self.mock_device_stream = Mock(name='device_stream')
        self.mock_stream.device_stream.return_value = self.mock_device_stream

        # Override the class attribute
        mstx.NPU_PROFILER = self.mock_profiler
        mstx.enable = True
        context.set_context(device_target="Ascend")

    def test_mark_should_call_profiler_mark_when_message_provided(self):
        """Should call profiler's mstx_mark method when message is provided."""
        message = "test_message"
        mstx.mark(message)
        self.mock_profiler.mstx_mark.assert_called_once_with(message, None, "default")
        mstx.mark(message, None, "domain")
        self.mock_profiler.mstx_mark.assert_called_with(message, None, "domain")

    def test_mark_should_call_profiler_mark_when_valid_stream_provided(self):
        """Should call profiler's mstx_mark method with device stream when valid stream is provided."""
        message = "test_message"
        mstx.mark(message, self.mock_stream)
        self.mock_profiler.mstx_mark.assert_called_once_with(message, self.mock_device_stream, "default")
        mstx.mark(message, self.mock_stream, "domain")
        self.mock_profiler.mstx_mark.assert_called_with(message, self.mock_device_stream, "domain")

    def test_range_start_should_return_range_id_when_message_provided(self):
        """Should return range ID when valid message is provided."""
        message = "test_range"
        range_id = mstx.range_start(message)
        self.mock_profiler.mstx_range_start.assert_called_once_with(message, None, "default")
        self.assertEqual(range_id, 1)
        mstx.range_start(message, None, "domain")
        self.mock_profiler.mstx_range_start.assert_called_with(message, None, "domain")

    def test_range_start_should_return_range_id_when_valid_stream_provided(self):
        """Should return range ID when valid message and stream are provided."""
        message = "test_range"
        range_id = mstx.range_start(message, self.mock_stream)
        self.mock_profiler.mstx_range_start.assert_called_once_with(message, self.mock_device_stream, "default")
        self.assertEqual(range_id, 1)
        mstx.range_start(message, self.mock_stream, "domain")
        self.mock_profiler.mstx_range_start.assert_called_with(message, self.mock_device_stream, "domain")

    def test_range_end_should_call_profiler_range_end_when_valid_id_provided(self):
        """Should call profiler's mstx_range_end method when valid range ID is provided."""
        range_id = 1
        mstx.range_end(range_id)
        self.mock_profiler.mstx_range_end.assert_called_once_with(range_id, "default")
        mstx.range_end(range_id, "domain")
        self.mock_profiler.mstx_range_end.assert_called_with(range_id, "domain")

    def test_mark_should_log_warning_when_invalid_stream_provided(self):
        """Should log a warning when an invalid stream is provided."""
        with patch('mindspore.log.warning') as mock_warning:
            mstx.mark("test_message", "invalid_stream")
            mock_warning.assert_called_once_with(
                "Invalid stream for mstx.mark func. Expected mindspore.runtime.Stream but got <class 'str'>."
            )
            self.mock_profiler.mstx_mark.assert_not_called()

    def test_range_start_should_log_warning_when_empty_message_provided(self):
        """Should log a warning when empty message is provided."""
        with patch('mindspore.log.warning') as mock_warning:
            range_id = mstx.range_start("")
            mock_warning.assert_called_once_with(
                "Invalid message for mstx.range_start func. Please input valid message string."
            )
            self.assertEqual(range_id, 0)
            self.mock_profiler.mstx_range_start.assert_not_called()

    def test_range_start_should_log_warning_when_invalid_stream_provided(self):
        """Should log a warning when invalid stream is provided."""
        with patch('mindspore.log.warning') as mock_warning:
            range_id = mstx.range_start("test_range", "invalid_stream")
            mock_warning.assert_called_once_with(
                "Invalid stream for mstx.range_start func. Expected mindspore.runtime.Stream but got <class 'str'>."
            )
            self.assertEqual(range_id, 0)
            self.mock_profiler.mstx_range_start.assert_not_called()

    def test_range_end_should_log_warning_when_invalid_range_id_provided(self):
        """Should log a warning when invalid range ID type is provided."""
        with patch('mindspore.log.warning') as mock_warning:
            mstx.range_end("invalid_id")
            mock_warning.assert_called_once_with(
                "Invalid range_id for mstx.range_end func. Please input return value from mstx.range_start."
            )
            self.mock_profiler.mstx_range_end.assert_not_called()

    def test_mark_shoule_return_when_mstx_not_enabled(self):
        """Should return 0 when mstx is not enabled."""
        mstx.enable = False
        mstx.mark('test_message')
        self.mock_profiler.mstx_mark.assert_not_called()

    def test_range_start_shoule_return_zero_when_mstx_not_enabled(self):
        """Should return 0 when mstx is not enabled."""
        mstx.enable = False
        range_id = mstx.range_start("test_range")
        self.assertEqual(range_id, 0)

    def test_range_end_shoule_return_when_mstx_not_enabled(self):
        """Should return 0 when mstx is not enabled."""
        mstx.enable = False
        mstx.range_end(1)
        self.mock_profiler.mstx_range_end.assert_not_called()

    def test_range_end_shoule_return_when_range_id_is_zero(self):
        """Should return 0 when mstx is not enabled."""
        mstx.range_end(0)
        self.mock_profiler.mstx_range_end.assert_not_called()


if __name__ == '__main__':
    unittest.main()
