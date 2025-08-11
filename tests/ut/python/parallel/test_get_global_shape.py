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
import pytest
from mindspore.parallel import Layout

def test_single_axis_single_shard():
    '''
    Feature: Single-axis single sharding.
    Description: Test scenario with one axis sharded once.
    Expectation: Global shape should be correctly computed from slice shape.
    '''
    layout = Layout(device_matrix=(2, 2, 2), alias_name=("dp", "sp", "mp"))
    layout_config = layout("dp", "sp", "mp")
    slice_shape = (3, 4, 5)

    assert layout_config.get_global_shape(slice_shape) == (6, 8, 10)

def test_single_axis_double_shard():
    '''
    Feature: Single-axis nested sharding.
    Description: Test scenario with one axis sharded twice (nested sharding).
    Expectation: Global shape should account for multiple sharding dimensions.
    '''
    layout = Layout(device_matrix=(2, 2, 2), alias_name=("dp", "sp", "mp"))
    layout_config = layout(("dp", "mp"), "sp", "None")
    slice_shape = (3, 4, 5)

    assert layout_config.get_global_shape(slice_shape) == (12, 8, 5)

def test_mixed_sharding():
    '''
    Feature: Mixed sharding strategy.
    Description: Test combination of single and double sharding on different axes.
    Expectation: Global shape should reflect compound sharding factors.
    '''
    layout = Layout(device_matrix=(2, 3, 4), alias_name=("data", "model", "pipeline"))
    layout_config = layout("data", ("model", "pipeline"), "None")
    slice_shape = (10, 20, 30)

    assert layout_config.get_global_shape(slice_shape) == (20, 240, 30)

def test_no_sharding():
    '''
    Feature: No sharding scenario.
    Description: Test unsharded tensor configuration.
    Expectation: Global shape should match local slice shape.
    '''
    layout = Layout(device_matrix=(2, 2), alias_name=("dp", "mp"))
    layout_config = layout("None", "None")
    slice_shape = (100, 200)
    assert layout_config.get_global_shape(slice_shape) == (100, 200)

def test_error_conditions():
    '''
    Feature: Error handling in layout configuration.
    Description: Test invalid inputs to layout methods.
    Expectation: Appropriate errors should be raised for invalid configurations.
    '''
    layout = Layout(device_matrix=(2, 2), alias_name=("dp", "mp"))

    with pytest.raises(ValueError, match="tensor_map is not set"):
        layout.get_global_shape((10, 10))

    layout_config = layout("dp", "None")
    with pytest.raises(ValueError, match="Length of slice_shape"):
        layout_config.get_global_shape((10,))
