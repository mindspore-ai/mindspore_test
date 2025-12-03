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

import pytest
from mindspore.parallel import Layout
from mindspore.parallel.spmd.ops.parallel_transpose import TransposeDistributedOp


class TestTransposeDistributedOp:
    """Test cases for TransposeDistributedOp.infer_layout method."""

    def test_basic_transpose_operation(self):
        """
        Feature: Transpose distributed operator basic functionality
        Description: Test transpose operation with valid 2D layout and axis permutation
        Expectation: Output layout tensor map correctly permuted according to axis
        """
        input_layout = Layout(device_matrix=[2, 2], alias_name="input")
        layout_with_tensor_map = input_layout(0, 1)
        
        op = TransposeDistributedOp()
        result_layout = op.infer_layout([layout_with_tensor_map], (1, 0))
        
        assert result_layout.tensor_map == (1, 0)
        assert result_layout.device_matrix == input_layout.device_matrix
        assert result_layout.alias_name == input_layout.alias_name

    def test_3d_transpose_operation(self):
        """
        Feature: Transpose distributed operator with 3D tensor
        Description: Test transpose operation with 3D layout and complex axis permutation
        Expectation: Output layout tensor map correctly follows the given permutation
        """
        input_layout = Layout(device_matrix=[2, 2, 2], alias_name="input_3d")
        layout_with_tensor_map = input_layout(0, 1, 2)
        
        op = TransposeDistributedOp()
        result_layout = op.infer_layout([layout_with_tensor_map], (2, 0, 1))
        
        assert result_layout.tensor_map == (2, 0, 1)
        assert result_layout.device_matrix == input_layout.device_matrix
        assert result_layout.alias_name == input_layout.alias_name

    def test_dimension_mismatch_error(self):
        """
        Feature: Transpose distributed operator error handling
        Description: Test transpose operation with mismatched tensor map and axis dimensions
        Expectation: Raise ValueError with appropriate message
        """
        input_layout = Layout(device_matrix=[2, 2], alias_name="mismatch_test")
        layout_with_tensor_map = input_layout(0, 1)
        
        op = TransposeDistributedOp()
        
        with pytest.raises(ValueError, match="Input tensor shape and permutation must have the same size"):
            op.infer_layout([layout_with_tensor_map], (1, 0, 2))

    def test_negative_index_error(self):
        """
        Feature: Transpose distributed operator validation
        Description: Test transpose operation with negative index in axis permutation
        Expectation: Raise ValueError for invalid permutation
        """
        input_layout = Layout(device_matrix=[2, 2, 2], alias_name="invalid_test")
        layout_with_tensor_map = input_layout(0, 1, 2)
        
        op = TransposeDistributedOp()
        
        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], (-1, 0, 1))

    def test_out_of_range_index_error(self):
        """
        Feature: Transpose distributed operator bounds checking
        Description: Test transpose operation with axis index exceeding tensor dimensions
        Expectation: Raise ValueError for invalid permutation due to out of range index
        """
        input_layout = Layout(device_matrix=[2, 2], alias_name="range_test")
        layout_with_tensor_map = input_layout(0, 1)
        
        op = TransposeDistributedOp()
        
        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], (0, 2))

    def test_duplicate_indices_error(self):
        """
        Feature: Transpose distributed operator uniqueness validation
        Description: Test transpose operation with duplicate indices in axis permutation
        Expectation: Raise ValueError for invalid permutation due to duplicate indices
        """
        input_layout = Layout(device_matrix=[2, 2, 2], alias_name="duplicate_test")
        layout_with_tensor_map = input_layout(0, 1, 2)
        
        op = TransposeDistributedOp()
        
        with pytest.raises(ValueError, match="Invalid permutation"):
            op.infer_layout([layout_with_tensor_map], (1, 1, 0))

    def test_identity_transpose(self):
        """
        Feature: Transpose distributed operator identity operation
        Description: Test transpose operation with identity permutation (no change)
        Expectation: Output layout identical to input layout
        """
        input_layout = Layout(device_matrix=[2, 2, 2], alias_name="identity_test")
        layout_with_tensor_map = input_layout(1, 0, 2)
        
        op = TransposeDistributedOp()
        result_layout = op.infer_layout([layout_with_tensor_map], (0, 1, 2))
        
        assert result_layout.tensor_map == (1, 0, 2)
        assert result_layout.device_matrix == input_layout.device_matrix
        assert result_layout.alias_name == input_layout.alias_name