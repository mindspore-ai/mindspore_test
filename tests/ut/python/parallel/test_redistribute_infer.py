# Copyright 2025 Huawei Technologies Co., Ltd
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

from mindspore.parallel.redistribute_infer import (
    RedistributionOperatorInfer,
    Status,
    CONCAT_BY_AXIS,
    SPLIT_BY_AXIS,
    PERMUTE_BY_AXIS,
    NONE,
    DevMat
)


class TestRedistributionOperatorInfer:
    def print_op_list(self, op_list):
        """print op list"""
        for op in op_list:
            if op[0] == 0:
                print("AllConcat:", op[1])
            if op[0] == 2:
                print("AlltoAll:", op[1])
            if op[0] == 1:
                print("AllSplit:", op[1])

    def test_simple_split_operation(self):
        """
        Feature: Test split operation
        Description: Test scenario requiring split operations
        Expectation: Should return SUCCESS with correct split operators
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4],
            in_tensor_map=[NONE, NONE],
            out_tensor_map=[0, 1]
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 2
        op1 = inferrer.operator_list_[0]
        op2 = inferrer.operator_list_[1]
        assert op1[0] == SPLIT_BY_AXIS
        assert op1[1] == (0, 0, 4)
        assert op2[0] == SPLIT_BY_AXIS
        assert op2[1] == (1, 1, 8)

    def test_simple_concat_operation(self):
        """
        Feature: Test concat operation
        Description: Test scenario requiring concat operations
        Expectation: Should return SUCCESS with correct concat operators
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4],
            in_tensor_map=[0, 1],
            out_tensor_map=[NONE, NONE]
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 2
        op1 = inferrer.operator_list_[0]
        op2 = inferrer.operator_list_[1]
        assert op1[0] == CONCAT_BY_AXIS
        assert op1[1] == (0, 0, 4)
        assert op2[0] == CONCAT_BY_AXIS
        assert op2[1] == (1, 1, 8)

    def test_permute_operation(self):
        """
        Feature: Test permute operation
        Description: Test dimension swapping with permute
        Expectation: Should return SUCCESS with correct permute operator
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4],
            in_tensor_map=[0, 1],
            out_tensor_map=[1, 0],
            use_permute=True
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 3
        op = inferrer.operator_list_[1]
        assert op[0] == PERMUTE_BY_AXIS
        assert op[1] == (8, 0, 1, 1, 8)

    def test_permute_with_large_dev_mat(self):
        """
        Feature: Test permute with large device matrix
        Description: Test permute operation in 3D device matrix
        Expectation: Should return SUCCESS with correct permute parameters
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[64, 32, 16],
            in_tensor_map=[0, 1, 2],
            out_tensor_map=[2, 0, 1],
            use_permute=True
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 4
        op = inferrer.operator_list_[1]
        assert op[0] == PERMUTE_BY_AXIS
        assert op[1] == (64, 0, 2, 2, 64)

    def test_permute_with_multiple_conflicts(self):
        """
        Feature: Test permute with multiple conflicts
        Description: Test permute with partial NONE mappings
        Expectation: Should return SUCCESS with correct permute sequence
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[32, 16, 8],
            in_tensor_map=[0, 1, NONE],
            out_tensor_map=[1, NONE, 0],
            use_permute=True
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 2
        op1, op2 = inferrer.operator_list_
        assert op1[0] == PERMUTE_BY_AXIS
        assert op1[1] == (8, 2, 0, 0, 8)
        assert op2[0] == PERMUTE_BY_AXIS
        assert op2[1] == (16, 0, 1, 1, 16)

    def test_mixed_operations(self):
        """
        Feature: Test mixed redistribution operations
        Description: Test combination of concat and split operations
        Expectation: Should return SUCCESS with correct operation sequence
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4, 2],
            in_tensor_map=[0, 1, NONE],
            out_tensor_map=[2, 0, 1],
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 3
        op1, op2, op3 = inferrer.operator_list_
        assert op1[0] == PERMUTE_BY_AXIS
        assert op1[1] == (4, 2, 1, 1, 4)
        assert op2[0] == PERMUTE_BY_AXIS
        assert op2[1] == (2, 1, 0, 0, 2)
        assert op3[0] == SPLIT_BY_AXIS
        assert op3[1] == (0, 2, 8)

    def test_complex_redistribution(self):
        """
        Feature: Test complex redistribution scenario
        Description: Test multi-operation redistribution sequence
        Expectation: Should return SUCCESS with correct operation list
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[16, 8, 4],
            in_tensor_map=[0, 1, NONE],
            out_tensor_map=[1, NONE, 0],
            use_permute=False
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 4
        op1, op2, op3, op4 = inferrer.operator_list_
        assert op1[0] == CONCAT_BY_AXIS
        assert op1[1] == (0, 0, 4)
        assert op2[0] == SPLIT_BY_AXIS
        assert op2[1] == (2, 0, 4)
        assert op3[0] == CONCAT_BY_AXIS
        assert op3[1] == (1, 1, 8)
        assert op4[0] == SPLIT_BY_AXIS
        assert op4[1] == (0, 1, 8)

    def test_special_concat_case(self):
        """
        Feature: Test special concat case
        Description: Test concat with partial no-change mappings
        Expectation: Should return SUCCESS with single split operation
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[0, 1]
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 1
        op1 = inferrer.operator_list_[0]
        assert op1[0] == SPLIT_BY_AXIS
        assert op1[1] == (1, 1, 8)

    def test_partial_mapping_change(self):
        """
        Feature: Test partial mapping change
        Description: Test when only part of mapping changes
        Expectation: Should return SUCCESS with single concat operation
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4],
            in_tensor_map=[0, 1],
            out_tensor_map=[0, NONE]
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 1
        op = inferrer.operator_list_[0]
        assert op[0] == CONCAT_BY_AXIS
        assert op[1] == (1, 1, 8)

    def test_none_to_none(self):
        """
        Feature: Test NONE to NONE mapping
        Description: Test when both input and output have NONE mappings
        Expectation: Should return SUCCESS with concat operation
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4],
            in_tensor_map=[0, NONE],
            out_tensor_map=[NONE, NONE]
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        assert len(inferrer.operator_list_) == 1
        op = inferrer.operator_list_[0]
        assert op[0] == CONCAT_BY_AXIS
        assert op[1] == (0, 0, 4)

    def test_tuple_shard_multi_operations(self):
        """
        Feature: Test mixed redistribution operations, tuple shard.
        Description: Test combination of concat and split operations
        Expectation: Should return SUCCESS with correct operation sequence
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4, 2],
            in_tensor_map=[(0, 1), NONE],
            out_tensor_map=[2, (0, 1)],
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        self.print_op_list(inferrer.operator_list_)
        assert len(inferrer.operator_list_) == 2
        op1, op2 = inferrer.operator_list_
        assert op1[0] == PERMUTE_BY_AXIS
        assert op1[1] == (8, 1, 0, (0, 1), 8)
        assert op2[0] == SPLIT_BY_AXIS
        assert op2[1] == (0, 2, 8)

    def test_tuple_shard_concat(self):
        """
        Feature: Test tuple shard concat operations
        Description: Test combination of concat and split operations
        Expectation: Should return SUCCESS with correct operation sequence
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4, 2],
            in_tensor_map=[(0, 1), 2],
            out_tensor_map=[NONE, 2],
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        self.print_op_list(inferrer.operator_list_)
        assert len(inferrer.operator_list_) == 1
        op1 = inferrer.operator_list_[0]
        assert op1[0] == CONCAT_BY_AXIS
        assert op1[1] == (0, (0, 1), 8)

    def test_tuple_shard_split(self):
        """
        Feature: Test tuple shard split operations
        Description: Test combination of concat and split operations
        Expectation: Should return SUCCESS with correct operation sequence
        """
        inferrer = RedistributionOperatorInfer(
            dev_mat=[8, 4, 2],
            in_tensor_map=[0, NONE],
            out_tensor_map=[0, (1, 2)],
        )
        status = inferrer.InferRedistributionOperator()
        assert status == Status.SUCCESS
        self.print_op_list(inferrer.operator_list_)
        assert len(inferrer.operator_list_) == 1
        op1 = inferrer.operator_list_[0]
        assert op1[0] == SPLIT_BY_AXIS
        assert op1[1] == (1, (1, 2), 32)

def test_2x3_single_dim():
    """
    Feature: Test rank_list generate by device matrix
    Description: Test rank_list generate
    Expectation: Should return SUCCESS with correct operation sequence
    """
    devmat = DevMat([2, 3])
    rank_list = [0, 1, 2, 3, 4, 5]

    # Test dimension 0 (slowest-changing dimension)
    assert devmat.GetDevicesAlongDim(0, rank_list, 0) == [0, 3]
    assert devmat.GetDevicesAlongDim(1, rank_list, 0) == [1, 4]
    assert devmat.GetDevicesAlongDim(2, rank_list, 0) == [2, 5]
    assert devmat.GetDevicesAlongDim(3, rank_list, 0) == [0, 3]  # Same group as rank0
    assert devmat.GetDevicesAlongDim(4, rank_list, 0) == [1, 4]  # Same group as rank1
    assert devmat.GetDevicesAlongDim(5, rank_list, 0) == [2, 5]  # Same group as rank2

    # Test dimension 1 (fastest-changing dimension)
    assert devmat.GetDevicesAlongDim(0, rank_list, 1) == [0, 1, 2]
    assert devmat.GetDevicesAlongDim(1, rank_list, 1) == [0, 1, 2]  # Same group as rank0
    assert devmat.GetDevicesAlongDim(2, rank_list, 1) == [0, 1, 2]  # Same group as rank0
    assert devmat.GetDevicesAlongDim(3, rank_list, 1) == [3, 4, 5]
    assert devmat.GetDevicesAlongDim(4, rank_list, 1) == [3, 4, 5]  # Same group as rank3
    assert devmat.GetDevicesAlongDim(5, rank_list, 1) == [3, 4, 5]  # Same group as rank3

def test_2x3x2_single_dim():
    """
    Feature: Test rank_list generate by device matrix
    Description: Test rank_list generate
    Expectation: Should return SUCCESS with correct operation sequence
    """
    devmat = DevMat([2, 3, 2])
    rank_list = list(range(12))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Test dimension 0 (slowest)
    assert devmat.GetDevicesAlongDim(0, rank_list, 0) == [0, 6]
    assert devmat.GetDevicesAlongDim(6, rank_list, 0) == [0, 6]  # Same group as rank0

    # Test dimension 1 (middle)
    assert devmat.GetDevicesAlongDim(0, rank_list, 1) == [0, 2, 4]
    assert devmat.GetDevicesAlongDim(2, rank_list, 1) == [0, 2, 4]  # Same group as rank0

    # Test dimension 2 (fastest)
    assert devmat.GetDevicesAlongDim(0, rank_list, 2) == [0, 1]
    assert devmat.GetDevicesAlongDim(1, rank_list, 2) == [0, 1]  # Same group as rank0

def test_combined_single_dim():
    """
    Feature: Test rank_list generate by device matrix
    Description: Test rank_list generate
    Expectation: Should return SUCCESS with correct operation sequence
    """
    devmat = DevMat([2, 3])
    rank_list = [0, 1, 2, 3, 4, 5]

    # Combined dimension with one element (equivalent to single dimension)
    assert devmat.GetDevicesAlongDim(0, rank_list, [0]) == [0, 3]
    assert devmat.GetDevicesAlongDim(0, rank_list, [1]) == [0, 1, 2]

def test_2x2_combined_dims():
    """
    Feature: Test rank_list generate by device matrix
    Description: Test rank_list generate
    Expectation: Should return SUCCESS with correct operation sequence
    """
    devmat = DevMat([2, 2])
    rank_list = [0, 1, 2, 3]

    # Combined dimensions (0, 1)
    assert devmat.GetDevicesAlongDim(0, rank_list, [0, 1]) == [0, 1, 2, 3]
    assert devmat.GetDevicesAlongDim(1, rank_list, [0, 1]) == [0, 1, 2, 3]
    assert devmat.GetDevicesAlongDim(2, rank_list, [0, 1]) == [0, 1, 2, 3]
    assert devmat.GetDevicesAlongDim(3, rank_list, [0, 1]) == [0, 1, 2, 3]

def test_2x3x2_combined_dims():
    """
    Feature: Test rank_list generate by device matrix
    Description: Test rank_list generate
    Expectation: Should return SUCCESS with correct operation sequence
    """
    devmat = DevMat([2, 3, 2])
    rank_list = list(range(12))

    # Combined dimensions (0, 1) for rank0
    assert devmat.GetDevicesAlongDim(0, rank_list, [0, 1]) == [0, 2, 4, 6, 8, 10]
    # Combined dimensions (1, 2) for rank0
    assert devmat.GetDevicesAlongDim(0, rank_list, [1, 2]) == [0, 1, 2, 3, 4, 5]
    # Combined dimensions (0, 2) for rank0
    assert devmat.GetDevicesAlongDim(0, rank_list, [0, 2]) == [0, 1, 6, 7]
