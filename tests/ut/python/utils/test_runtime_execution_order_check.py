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
""" test_runtime_execution_order_check """
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open

from mindspore.utils.runtime_execution_order_check import (
    RankFolderParser,
    modify_execute_orders,
    ExecuteOrder,
    parse_and_validate,
    detect_cycle_in_graph,
    comm_exec_order_check,
)

class TestRankFolderParser(unittest.TestCase):
    def setUp(self):
        # Create temporary directories and files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rank_folder_1 = os.path.join(self.temp_dir.name, 'rank_0/execute_order')
        os.makedirs(self.rank_folder_1)
        self.execute_order_file_1 = os.path.join(self.rank_folder_1, 'comm_execute_order.csv')
        with open(self.execute_order_file_1, 'w') as file:
            file.write(
                'index,group,comm_rank,primitive,src_rank,dest_rank,root_rank,input_shape,input_type,'
                'output_shape,output_type,input_size,output_size\n')
            file.write('2,A,1,primitive,0,1,0,shape1,type1,shape2,type2,size1,size2\n')

        self.rank_folder_2 = os.path.join(self.temp_dir.name, 'rank_1/execute_order')
        os.makedirs(self.rank_folder_2)
        self.execute_order_file_2 = os.path.join(self.rank_folder_2, 'comm_execute_order.csv')
        with open(self.execute_order_file_2, 'w') as file:
            file.write(
                'index,group,comm_rank,primitive,src_rank,dest_rank,root_rank,input_shape,input_type,'
                'output_shape,output_type,input_size,output_size\n')
            file.write('3,B,2,primitive,1,0,1,shape3,type3,shape4,type4,size3,size4\n')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_paths_valid_directory(self):
        parser = RankFolderParser(folders=self.temp_dir.name)
        self.assertEqual(parser.folders, [self.temp_dir.name])

    def test_validate_paths_invalid_path(self):
        # Test invalid path handling
        with self.assertRaises(ValueError) as context:
            RankFolderParser(folders='/nonexistent/path/to/folder')
        self.assertIn("Path does not exist", str(context.exception))

    def test_validate_paths_missing_rank_folder(self):
        # Test a directory without rank_x folder
        new_temp_dir = tempfile.TemporaryDirectory()
        with self.assertRaises(ValueError) as context:
            RankFolderParser(folders=new_temp_dir.name)
        self.assertIn("No rank_x folders found", str(context.exception))
        new_temp_dir.cleanup()

    def test_parse_multiple_rank_folders(self):
        # Test parsing multiple rank_{x} folders
        parser = RankFolderParser(folders=[self.temp_dir.name])
        result = parser.parse()
        self.assertEqual(len(result), 2)
        self.assertIn('0', result)
        self.assertIn('1', result)
        self.assertEqual(len(result['0']), 1)
        self.assertEqual(len(result['1']), 1)

    @patch("builtins.open", new_callable=mock_open,
           read_data="index,group,comm_rank,primitive,src_rank,dest_rank,root_rank\n1,2,A,1,primitive,0,1,0\n")
    def test_invalid_header(self, mock_file):
        # Test file with an invalid header
        rank_folder_invalid = os.path.join(self.temp_dir.name, 'rank_invalid')
        os.makedirs(rank_folder_invalid)
        parser = RankFolderParser(folders=[rank_folder_invalid])

        with patch("os.path.exists", return_value=True), patch("os.path.isdir", return_value=True):
            result = parser.parse_rank_folder(rank_folder_invalid, 'invalid')
            self.assertIsNone(result[1])

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_missing_execute_order_file(self, mock_open_fn):
        # Test missing execute_order.csv file handling
        rank_folder_missing_file = os.path.join(self.temp_dir.name, 'rank_missing')
        os.makedirs(rank_folder_missing_file)
        parser = RankFolderParser(folders=[rank_folder_missing_file])

        result = parser.parse_rank_folder(rank_folder_missing_file, 'missing')
        self.assertIsNone(result[1])


class TestModifyExecuteOrders(unittest.TestCase):

    def setUp(self):
        # Prepare sample ExecuteOrder objects for different scenarios

        # Scenario: allGather (collective communication)
        self.all_gather_order = ExecuteOrder(
            index="1", group="A", comm_rank="0 1 2 3", primitive="allGather"
        )

        # Scenario: Send operation
        self.send_order = ExecuteOrder(
            index="2", group="B", comm_rank="0", primitive="Send", src_rank="0", dest_rank="1",
            input_shape="shape1"
        )

        # Scenario: Receive operation
        self.receive_order = ExecuteOrder(
            index="3", group="B", comm_rank="0", primitive="Receive", src_rank="0", dest_rank="1",
            output_shape="shape2"
        )

        # Scenario: Broadcast with root_rank
        self.broadcast_order = ExecuteOrder(
            index="4", group="C", comm_rank="0 1 2 3", primitive="broadcast", root_rank="0"
        )

        # Scenario: Send without input_shape (edge case)
        self.send_no_input_shape_order = ExecuteOrder(
            index="5", group="D", comm_rank="0", primitive="Send", src_rank="0", dest_rank="1"
        )

    def test_all_gather_order(self):
        # Test the allGather operation key generation
        execute_orders_map = {
            "0": [self.all_gather_order]
        }
        result = modify_execute_orders(execute_orders_map)

        expected_key = "allGather_A_(0,1,2,3)_1th"
        self.assertIn("0", result)
        self.assertEqual(result["0"][0], expected_key)

    def test_send_order(self):
        # Test the Send operation key generation
        execute_orders_map = {
            "0": [self.send_order]
        }
        result = modify_execute_orders(execute_orders_map)

        expected_key = "Send_Receive_B_(0)->(1)_shape1_1th"
        self.assertIn("0", result)
        self.assertEqual(result["0"][0], expected_key)

    def test_receive_order(self):
        # Test the operation key generation
        execute_orders_map = {
            "1": [self.receive_order]
        }
        result = modify_execute_orders(execute_orders_map)

        expected_key = "Send_Receive_B_(0)->(1)_shape2_1th"
        self.assertIn("1", result)
        self.assertEqual(result["1"][0], expected_key)

    def test_broadcast_order(self):
        # Test the broadcast operation with root_rank key generation
        execute_orders_map = {
            "0": [self.broadcast_order]
        }
        result = modify_execute_orders(execute_orders_map)

        expected_key = "broadcast_C_commRank:(0,1,2,3)_rootRank:(0)_1th"
        self.assertIn("0", result)
        self.assertEqual(result["0"][0], expected_key)

    def test_multiple_orders_same_rank(self):
        # Test multiple ExecuteOrder instances with the same rank
        execute_orders_map = {
            "0": [self.send_order, self.receive_order, self.broadcast_order, self.all_gather_order]
        }
        result = modify_execute_orders(execute_orders_map)

        expected_keys = [
            "Send_Receive_B_(0)->(1)_shape1_1th",
            "Send_Receive_B_(0)->(0)_shape2_1th",
            "broadcast_C_commRank:(0,1,2,3)_rootRank:(0)_1th",
            "allGather_A_(0,1,2,3)_1th"
        ]
        self.assertIn("0", result)
        self.assertEqual(result["0"], expected_keys)

    def test_multiple_instances_of_same_order(self):
        # Test multiple instances of the same order to validate count handling
        execute_orders_map = {
            "0": [self.send_order, self.send_order]
        }
        result = modify_execute_orders(execute_orders_map)

        expected_keys = [
            "Send_Receive_B_(0)->(1)_shape1_1th",
            "Send_Receive_B_(0)->(1)_shape1_2th"
        ]
        self.assertIn("0", result)
        self.assertEqual(result["0"], expected_keys)


class TestParseAndValidate(unittest.TestCase):
    def test_invalid_value_type(self):
        data = {
            "1": "not_a_list",
            "2": [1, 2, 3],
            "3": ["valid", "list", "of_strings"]
        }
        parse_and_validate(data)

    def test_valid_input(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_1th"],
            "2": ["Send_Receive_(1)->(2)_1th"]
        }
        parse_and_validate(data)

    def test_missing_keys_all_rank_true(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_1th"]
        }
        parse_and_validate(data, all_rank=True)

    def test_value_missing_in_referenced_keys(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_1th"],
            "2": ["Send_Receive_(1)->(2)_1th", "Send_Receive_(1)->(2)_2th"]
        }
        parse_and_validate(data)

    def test_empty_values(self):
        data = {
            "1": [],
            "2": [""]
        }
        parse_and_validate(data)

    def test_complex_nesting_and_cross_references(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_1th", "allGather_(1,3)_1th"],
            "2": ["Send_Receive_(1)->(2)_1th"],
            "3": ["allGather_(1,3)_1th"]
        }
        parse_and_validate(data)


class TestDetectCycleInGraph(unittest.TestCase):

    def test_cycle_in_graph(self):
        ranks_map = {
            '1': ['A', 'B', 'C', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'B', 'C', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> B', '1 B -> C', '1 C -> A'])

    def test_no_cycle_in_graph(self):
        ranks_map = {
            'r1': ['A', 'B', 'C'],
            'r2': ['D', 'E']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

    def test_multiple_cycles(self):
        # Graph with two different cycles
        ranks_map = {
            '1': ['A', 'B', 'C', 'A'],
            '2': ['D', 'E', 'D']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertTrue(cycle_path in [['A', 'B', 'C', 'A'], ['D', 'E', 'D']])
        if cycle_path == ['A', 'B', 'C', 'A']:
            self.assertEqual(cycle_ranks, ['1 A -> B', '1 B -> C', '1 C -> A'])
        elif cycle_path == ['D', 'E', 'D']:
            self.assertEqual(cycle_ranks, ['2 D -> E', '2 E -> D'])

    def test_empty_graph(self):
        ranks_map = {}
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

    def test_two_nodes_no_cycle(self):
        ranks_map = {
            'rank1': ['A', 'B']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

    def test_complex_cycle(self):
        ranks_map = {
            "1": ["A", "B", "C"],
            "2": ["C", "D", "E"],
            "3": ["E", "A"]
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ["A", "B", "C", "D", "E", "A"])
        self.assertEqual(cycle_ranks, [
            '1 A -> B',
            '1 B -> C',
            '2 C -> D',
            '2 D -> E',
            '3 E -> A'
        ])

    def test_disconnected_components_with_cycle(self):
        ranks_map = {
            "1": ["A", "B", "C"],
            "2": ["D", "E", "D"]
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ["D", "E", "D"])
        self.assertEqual(cycle_ranks, ['2 D -> E', '2 E -> D'])

    def test_cross_rank_cycle(self):
        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'B', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> B', '2 B -> A'])

        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'C'],
            '3': ['C', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'B', 'C', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> B', '2 B -> C', '3 C -> A'])

        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'C'],
            '3': ['C', 'D']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

        ranks_map = {
            '1': ['A', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> A'])

        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'C'],
            '3': ['C', 'B']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['B', 'C', 'B'])
        self.assertEqual(cycle_ranks, ['2 B -> C', '3 C -> B'])

        ranks_map = {
            '1': ['A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

        ranks_map = {
            '1': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '4': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '5': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '6': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '7': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '8': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)


class TestCommExecOrderCheck(unittest.TestCase):
    """Test cases for CommExecOrderCheck class validation logic."""

    @patch('mindspore.log.error')
    def test_end_before_start(self, mock_logger):
        """Test calling end action before start."""
        with patch('mindspore._c_expression.CommExecOrderChecker.get_instance') as mock_get:
            mock_instance = mock_get.return_value
            comm_exec_order_check("end")
            mock_instance.stop_collect_exec_order.assert_not_called()
            mock_logger.assert_called_once_with("The 'end' action cannot be called before the 'start' action.")

    def test_normal_sequence(self):
        """Test correct start-end sequence."""
        with patch('mindspore._c_expression.CommExecOrderChecker.get_instance') as mock_get:
            mock_instance = mock_get.return_value
            comm_exec_order_check("start")
            comm_exec_order_check("end")

            mock_instance.start_collect_exec_order.assert_called_once()
            mock_instance.stop_collect_exec_order.assert_called_once()

    @patch('mindspore.log.error')
    def test_multiple_starts(self, mock_logger):
        """Test multiple start calls."""
        with patch('mindspore._c_expression.CommExecOrderChecker.get_instance') as mock_get:
            mock_instance = mock_get.return_value
            comm_exec_order_check("start")
            comm_exec_order_check("start")

            self.assertEqual(mock_instance.start_collect_exec_order.call_count, 1)
            self.assertEqual(mock_logger.call_count, 1)

    @patch('mindspore.log.error')
    def test_multiple_ends(self, mock_logger):
        """Test multiple end calls."""
        with patch('mindspore._c_expression.CommExecOrderChecker.get_instance') as mock_get:
            mock_instance = mock_get.return_value
            comm_exec_order_check("start")
            comm_exec_order_check("end")
            comm_exec_order_check("end")

            mock_instance.stop_collect_exec_order.assert_called_once()
            self.assertEqual(mock_logger.call_count, 1)


if __name__ == "__main__":
    unittest.main()
