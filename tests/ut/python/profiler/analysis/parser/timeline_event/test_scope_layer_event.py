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

from decimal import Decimal
import unittest

from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_event.scope_layer_event import (
    ScopeLayerCompleteEvent,
    ScopeLayerMetaEvent
)


class TestScopeLayerCompleteEvent(unittest.TestCase):
    """Test cases for ScopeLayerCompleteEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_scope",
            "dur": 500
        }
        self.event = ScopeLayerCompleteEvent(self.event_data)

    def test_pid_should_return_scope_layer_pid_when_accessed(self):
        """Should return correct scope layer process ID."""
        self.assertEqual(self.event.pid, int(EventConstant.SCOPE_LAYER_PID))

    def test_duration_should_be_settable_when_modified(self):
        """Should allow modifying duration after initialization."""
        new_duration = Decimal('1000')
        self.event.dur = new_duration
        self.assertEqual(self.event.dur, new_duration)


class TestScopeLayerMetaEvent(unittest.TestCase):
    """Test cases for ScopeLayerMetaEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_scope_meta"
        }
        self.event = ScopeLayerMetaEvent(self.event_data)

    def test_pid_should_return_scope_layer_pid_when_accessed(self):
        """Should return correct scope layer process ID."""
        self.assertEqual(self.event.pid, int(EventConstant.SCOPE_LAYER_PID))


if __name__ == '__main__':
    unittest.main()
