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
"""test ascend step trace time viewer"""
import unittest

import numpy as np
from unittest import mock
from decimal import Decimal

from mindspore.profiler.analysis.viewer.ascend_step_trace_time_viewer import AscendStepTraceTimeViewer
from mindspore.profiler.common.constant import ProfilerActivity


# pylint: disable=protected-access
class TestAscendStepTraceTimeViewer(unittest.TestCase):
    def setUp(self):
        self.viewer = AscendStepTraceTimeViewer(
            ascend_profiler_output_path="output_path",
            profiler_level="Level1",
            ascend_ms_dir="ascend_ms_dir"
        )

        self.times = np.array([
            (Decimal(5.0), Decimal(5.0)),
            (Decimal(10.0), Decimal(10.0)),
            (Decimal(30.0), Decimal(10.0)),
            (Decimal(45.0), Decimal(5.0))
        ], dtype=[('ts', object), ('dur', object)])

        self.step_id_to_time_dict = {
            0: (Decimal('2'), Decimal('20')),
            1: (Decimal('20'), Decimal('40')),
            2: (Decimal('40'), Decimal('50'))
        }
        self.viewer.trace_container = mock.MagicMock()
        self.viewer.computing_np = self.times
        self.viewer.communication_np = self.times
        self.viewer.communication_not_overlapped_np = self.times
        self.viewer.free_np = self.times
        self.viewer._activities = [ProfilerActivity.CPU.value]

    def test_calculate_event_total_time_by_step_should_return_total_time_when_input_correct_data(self):
        ts = Decimal('0')
        es = Decimal('20')
        result = self.viewer._calculate_event_total_time_by_step(self.times, ts, es)
        self.assertEqual(result, Decimal('15.0'))

        ts = Decimal('20')
        es = Decimal('40')
        result = self.viewer._calculate_event_total_time_by_step(self.times, ts, es)
        self.assertEqual(result, Decimal('10.0'))

        ts = Decimal('40')
        es = Decimal('50')
        result = self.viewer._calculate_event_total_time_by_step(self.times, ts, es)
        self.assertEqual(result, Decimal('5.0'))

    def test_calculate_free_event_total_time_by_step_should_return_total_time_when_input_correct_data(self):
        ts = Decimal('6')
        es = Decimal('12')
        result = self.viewer._calculate_free_event_total_time_by_step(self.times, ts, es)
        self.assertEqual(result, Decimal('6.0'))

        ts = Decimal('32')
        es = Decimal('42')
        result = self.viewer._calculate_free_event_total_time_by_step(self.times, ts, es)
        self.assertEqual(result, Decimal('8.0'))

        ts = Decimal('46')
        es = Decimal('52')
        result = self.viewer._calculate_free_event_total_time_by_step(self.times, ts, es)
        self.assertEqual(result, Decimal('4.0'))

    def test_calculate_event_total_time_by_step_should_return_first_time_when_input_correct_data(self):
        ts = Decimal('2')
        result = self.viewer._calculate_event_first_time_by_step(self.times, ts)
        self.assertEqual(result, Decimal('5.0'))

        ts = Decimal('20')
        result = self.viewer._calculate_event_first_time_by_step(self.times, ts)
        self.assertEqual(result, Decimal('30.0'))

        ts = Decimal('40')
        result = self.viewer._calculate_event_first_time_by_step(self.times, ts)
        self.assertEqual(result, Decimal('45.0'))

    def test_generate_step_trace_time_data(self):
        self.viewer.generate_step_trace_time_data(self.step_id_to_time_dict)

        expected_data = [
            {"Step": 0, "Computing": Decimal('15.0'), "Communication": Decimal('15.0'),
             "Communication(Not Overlapped)": Decimal('15.0'), "Free": Decimal('15.0'),
             "Overlapped": Decimal('0.0'), "Stage": Decimal('0'), "Bubble": Decimal('0'),
             "Communication(Not Overlapped and Exclude Receive)": Decimal('15.0'), "Preparing": Decimal('3.0')},
            {"Step": 1, "Computing": Decimal('10.0'), "Communication": Decimal('10.0'),
             "Communication(Not Overlapped)": Decimal('10.0'), "Free": Decimal('10.0'),
             "Overlapped": Decimal('0.0'), "Stage": Decimal('0'), "Bubble": Decimal('0'),
             "Communication(Not Overlapped and Exclude Receive)": Decimal('10.0'), "Preparing": Decimal('10.0')},
            {"Step": 2, "Computing": Decimal('5.0'), "Communication": Decimal('5.0'),
             "Communication(Not Overlapped)": Decimal('5.0'), "Free": Decimal('5.0'),
             "Overlapped": Decimal('0.0'), "Stage": Decimal('0'), "Bubble": Decimal('0'),
             "Communication(Not Overlapped and Exclude Receive)": Decimal('5.0'), "Preparing": Decimal('5.0')}
        ]

        self.assertEqual(self.viewer.step_trace_time_data_list, expected_data)


if __name__ == "__main__":
    unittest.main()
