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
"""Test the AscendCommunicationViewer class."""
import unittest
from unittest.mock import patch, MagicMock
import os

from mindspore.profiler.analysis.viewer.ascend_communication_viewer import AscendCommunicationViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.constant import JitLevel


# pylint: disable=protected-access
class TestAscendCommunicationViewer(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "ascend_profiler_output_path": "test_ascend_profiler_output_path",
            "msprof_analyze_output_path": "test_msprof_analyze_output_path",
            "ascend_ms_dir": "test_ascend_ms_dir",
            "is_set_schedule": True,
            "jit_level": JitLevel.GRAPH_LEVEL
        }
        self.viewer = AscendCommunicationViewer(**self.kwargs)
        self.step_list = [
            {"step_id": "1", "start_ts": 0, "end_ts": 100, "comm_ops": {}},
            {"step_id": "2", "start_ts": 100, "end_ts": 200, "comm_ops": {}}
        ]
        self.mock_comm_data = {
            "Send_Op1": {"Communication Time Info": {"Start Timestamp(us)": 50}},
            "Recv_Op2": {"Communication Time Info": {"Start Timestamp(us)": 150}},
            "AllReduce_Op3": {"Communication Time Info": {"Start Timestamp(us)": 250}}
        }
        self.mock_matrix_data = {
            "Send_Op1": [{"Bandwidth(GB/s)": 10, "Transport Type": "PCIE"}],
            "AllReduce_Op3": [{"Bandwidth(GB/s)": 20, "Transport Type": "NVL"}]
        }

    @patch.object(AscendCommunicationViewer, '_update_default_step_list')
    @patch.object(AscendCommunicationViewer, '_update_step_list')
    def test_init_step_list_should_success_when_correct(self, mock_update_step_list, mock_update_default_step_list):
        data = {"trace_view_container": MagicMock()}
        data["trace_view_container"].get_step_id_time_dict.return_value = {}
        self.viewer._is_set_schedule = False
        self.viewer._init_step_list(data)
        mock_update_default_step_list.assert_called_once()
        mock_update_step_list.assert_not_called()

        data["trace_view_container"].get_step_id_time_dict.return_value = {"1": (0, 10)}
        self.viewer._is_set_schedule = True
        self.viewer._jit_level = JitLevel.KBK_LEVEL
        self.viewer._init_step_list(data)
        mock_update_step_list.assert_called_once_with({"1": (0, 10)})

    def test_update_default_step_list_should_success_when_correct(self):
        self.viewer._update_default_step_list()
        expected = [{"step_id": "0", "start_ts": 0, "end_ts": float('inf'), "comm_ops": {}}]
        self.assertEqual(self.viewer.step_list, expected)

    def test_update_step_list_should_success_when_correct(self):
        step_id_to_time_dict = {"1": (0, 10), "2": (10, 20)}
        self.viewer._update_step_list(step_id_to_time_dict)
        expected = [
            {"step_id": "1", "start_ts": 0, "end_ts": 10, "comm_ops": {}},
            {"step_id": "2", "start_ts": 10, "end_ts": 20, "comm_ops": {}}
        ]
        self.assertEqual(self.viewer.step_list, expected)

    def test_generate_communication_should_success_when_correct(self):
        with patch.object(FileManager, 'read_json_file') as mock_read, \
             patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            mock_read.side_effect = [self.mock_comm_data, self.mock_comm_data]

            self.viewer.step_list = []
            self.viewer._generate_communication()
            self.assertEqual(self.viewer.output_communication, {})

            self.viewer.step_list = self.step_list
            self.viewer._generate_communication()
            self.assertIn("step1", self.viewer.output_communication)
            self.assertEqual(len(self.viewer.output_communication), 2)

    @patch.object(os.path, 'exists')
    @patch.object(FileManager, 'read_json_file')
    @patch.object(AscendCommunicationViewer, '_split_matrix_by_step')
    @patch.object(AscendCommunicationViewer, '_get_matrix_ops_dict')
    def test_generate_matrix_should_success_when_correct(
            self, mock_get_matrix_ops_dict,
            mock_split_matrix_by_step, mock_read_json_file, mock_exists
        ):
        mock_exists.return_value = True
        mock_read_json_file.return_value = {"test": "data"}
        self.viewer._generate_matrix()

        mock_split_matrix_by_step.assert_called_once_with({"test": "data"})
        matrix_data_by_step = mock_split_matrix_by_step.return_value
        for _, comm_matrix_data in matrix_data_by_step.items():
            mock_get_matrix_ops_dict.assert_any_call(comm_matrix_data)

    def test_split_comm_op_by_step_should_success_when_correct(self):
        """Test step-based communication op splitting"""
        self.viewer.step_list = self.step_list
        self.viewer._split_comm_op_by_step(self.mock_comm_data)

        self.assertEqual(list(self.step_list[0]["comm_ops"].keys()), ["Send_Op1"])
        self.assertEqual(list(self.step_list[1]["comm_ops"].keys()), ["Recv_Op2"])
        self.assertNotIn("AllReduce_Op3", self.step_list[0]["comm_ops"])
        self.assertNotIn("AllReduce_Op3", self.step_list[1]["comm_ops"])

    def test_split_communication_p2p_ops_should_success_when_correct(self):
        """Test P2P operation detection"""
        test_data = {
            "HCOM_Send_0": {},
            "Receive_1": {},
            "AllReduce_2": {}
        }
        p2p, collective = self.viewer._split_communication_p2p_ops(test_data).values()

        self.assertEqual(len(p2p), 2)
        self.assertEqual(len(collective), 1)
        self.assertIn("HCOM_Send_0", p2p)
        self.assertIn("AllReduce_2", collective)

    def test_split_matrix_by_step_should_success_when_correct(self):
        """Test matrix data splitting with mock step data"""
        self.viewer.step_list = [
            {"step_id": "1", "comm_ops": {"Send_Op1": ...}},
            {"step_id": "2", "comm_ops": {"Recv_Op2": ...}}
        ]
        result = self.viewer._split_matrix_by_step(self.mock_matrix_data)

        self.assertEqual(len(result["step1"]), 1)
        self.assertEqual(result["step1"]["Send_Op1"][0]["Bandwidth(GB/s)"], 10)
        self.assertEqual(result.get("step2", None), None)

    def test_get_communication_ops_dict_should_success_when_correct(self):
        """Test communication op classification"""
        test_data = {
            "Send_Op1": {},
            "Recv_Op2": {},
            "AllReduce_Op3": {}
        }
        result = self.viewer._get_communication_ops_dict(test_data)

        self.assertEqual(len(result["p2p"]), 2)
        self.assertIn("Send_Op1", result["p2p"])
        self.assertIn("AllReduce_Op3", result["collective"])

    def test_integrate_matrix_data_should_success_when_correct(self):
        test_data = {
            ("send", "op1", "link1"): [
                {self.viewer.BANDWIDTH_GB_S: 30,
                 self.viewer.TRANSPORT_TYPE: "PCIE",
                 "Transit Size(MB)": 100,
                 "Transit Time(ms)": 10},
                {self.viewer.BANDWIDTH_GB_S: 20,
                 self.viewer.TRANSPORT_TYPE: "PCIE",
                 "Transit Size(MB)": 200,
                 "Transit Time(ms)": 20},
                {self.viewer.BANDWIDTH_GB_S: 10,
                 self.viewer.TRANSPORT_TYPE: "PCIE",
                 "Transit Size(MB)": 300,
                 "Transit Time(ms)": 30}
            ],
            ("receive", "op2", "link2"): [
                {self.viewer.BANDWIDTH_GB_S: 50,
                 self.viewer.TRANSPORT_TYPE: "NVL",
                 "Transit Size(MB)": 50,
                 "Transit Time(ms)": 1}
            ]
        }
        result = self.viewer._integrate_matrix_data(test_data)
        send_key = "send-top1@op1"
        self.assertEqual(result[send_key]["link1"][self.viewer.BANDWIDTH_GB_S], 30)
        receive_key = "receive-middle@op2"
        self.assertEqual(result[receive_key]["link2"][self.viewer.BANDWIDTH_GB_S], 50)
        total_key = "send-total@op1"
        self.assertAlmostEqual(result[total_key]["link1"][self.viewer.BANDWIDTH_GB_S], 10.0, places=2)
        self.assertEqual(result[send_key]["link1"][self.viewer.TRANSPORT_TYPE], "PCIE")

    def test_get_matrix_ops_dict_should_success_when_correct(self):
        test_op_data = {
            "HCOM_Send_0@rank1": {
                "link1": {"Bandwidth(GB/s)": 10, "Transit Size(MB)": 100},
                "link2": {"Bandwidth(GB/s)": 20, "Transit Size(MB)": 200}
            },
            "HCOM_Receive_1@rank2": {
                "link3": {"Bandwidth(GB/s)": 30, "Transit Size(MB)": 300}
            },
            "AllReduce-op123@rank3": {
                "link4": {"Bandwidth(GB/s)": 40, "Transit Size(MB)": 400}
            }
        }
        result = self.viewer._get_matrix_ops_dict(test_op_data)

        p2p_ops = result[self.viewer.P2P]
        self.assertIn("send-top1@rank1", p2p_ops)
        self.assertEqual(p2p_ops["send-top1@rank1"]["link1"]["Bandwidth(GB/s)"], 10)
        self.assertIn("receive-middle@rank2", p2p_ops)
        collective_ops = result[self.viewer.COLLECTIVE]
        self.assertIn("allreduce-total@rank3", collective_ops)

    def test_compute_total_info_should_success_when_correct(self):
        test_comm_ops = {
            "Send-op1": {
                "Communication Time Info": {
                    "Wait Time(ms)": 100,
                    "Transit Time(ms)": 200,
                    "Synchronization Time(ms)": 50
                },
                "Communication Bandwidth Info": {
                    "PCIE": {
                        "Transit Size(MB)": 500,
                        "Transit Time(ms)": 100
                    }
                }
            },
            "AllReduce-op2": {
                "Communication Time Info": {
                    "Wait Time(ms)": 200,
                    "Transit Time(ms)": 300,
                    "Synchronization Time(ms)": 100
                },
                "Communication Bandwidth Info": {
                    "HCCS": {
                        "Transit Size(MB)": 1000,
                        "Transit Time(ms)": 200
                    }
                }
            }
        }
        self.viewer._compute_total_info(test_comm_ops)
        total_info = test_comm_ops['Total Op Info']
        time_info = total_info[self.viewer.COMMUNICATION_TIME_INFO]
        self.assertEqual(time_info["Wait Time(ms)"], 300)
        self.assertEqual(time_info["Transit Time(ms)"], 500)
        bandwidth_info = total_info[self.viewer.COMMUNICATION_BANDWIDTH_INFO]
        self.assertEqual(bandwidth_info["PCIE"]["Transit Size(MB)"], 500)
        self.assertEqual(bandwidth_info["HCCS"]["Transit Size(MB)"], 1000)
        self.assertAlmostEqual(time_info["Wait Time Ratio"], 0.375, places=3)  # 300/(300+500)
        self.assertAlmostEqual(time_info["Synchronization Time Ratio"], 0.23, places=2)  # 150/(500+150)

    def test_combine_bandwidth_info_should_success_when_correct(self):
        test_com_info = {
            "PCIE": {
                "Transit Time(ms)": 100,
                "Transit Size(MB)": 500,
                "Size Distribution": {
                    "128": [2, 200],
                    "256": [3, 300]
                }
            },
            "HCCS": {
                "Transit Time(ms)": 200,
                "Transit Size(MB)": 1000,
                "Size Distribution": {
                    "512": [5, 500]
                }
            }
        }
        total_dict = {}
        self.viewer._combine_bandwidth_info(test_com_info, total_dict)

        pcie_data = total_dict["PCIE"]
        self.assertEqual(pcie_data["Transit Time(ms)"], 100)
        self.assertEqual(pcie_data["Transit Size(MB)"], 500)
        self.assertEqual(pcie_data["Size Distribution"]["128"], [2, 200])
        self.assertEqual(pcie_data["Size Distribution"]["256"], [3, 300])

        hccs_data = total_dict["HCCS"]
        self.assertEqual(hccs_data["Transit Time(ms)"], 200)
        self.assertEqual(hccs_data["Transit Size(MB)"], 1000)
        self.assertEqual(hccs_data["Size Distribution"]["512"], [5, 500])

        test_com_info["NVL"] = {"Transit Time(ms)": 50, "Transit Size(MB)": 200}
        self.viewer._combine_bandwidth_info(test_com_info, total_dict)
        self.assertEqual(total_dict["NVL"]["Transit Time(ms)"], 50)
        self.assertEqual(total_dict["NVL"]["Transit Size(MB)"], 200)


if __name__ == "__main__":
    unittest.main()
