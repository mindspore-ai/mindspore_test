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

from mindspore.parallel._parallel_serialization import _get_param_list_when_first_dim_sharded


def test_get_param_list_when_first_dim_sharded_1():
    """
    Feature: test _get_param_list_when_first_dim_sharded.
    Description: 4 rank, first dim sharded 2 with device arrangement index 1.
    Expectation: Correct param list.
    """
    device_arrange = [2, 2]
    first_dim_sharded_device_index = 1
    rank = 0
    ret = _get_param_list_when_first_dim_sharded(device_arrange, first_dim_sharded_device_index, rank)
    assert ret == [0, 1]


def test_get_param_list_when_first_dim_sharded_2():
    """
    Feature: test _get_param_list_when_first_dim_sharded.
    Description: 8 rank, first dim sharded 2 with device arrangement index 2.
    Expectation: Correct param list.
    """
    device_arrange = [2, 2, 2]
    first_dim_sharded_device_index = 2
    rank = 0
    ret = _get_param_list_when_first_dim_sharded(device_arrange, first_dim_sharded_device_index, rank)
    assert ret == [0, 1, 2, 3]


def test_get_param_list_when_first_dim_sharded_3():
    """
    Feature: test _get_param_list_when_first_dim_sharded.
    Description: 8 rank, first dim sharded 2 with device arrangement index 0.
    Expectation: Correct param list.
    """
    device_arrange = [2, 2, 2]
    first_dim_sharded_device_index = 0
    rank = 5
    ret = _get_param_list_when_first_dim_sharded(device_arrange, first_dim_sharded_device_index, rank)
    assert ret == [0, 1, 2, 3, 4, 5, 6, 7]


def test_get_param_list_when_first_dim_sharded_4():
    """
    Feature: test _get_param_list_when_first_dim_sharded.
    Description: 8 rank, first dim sharded 2 with device arrangement index 0.
    Expectation: Correct param list.
    """
    device_arrange = [64, 8]
    first_dim_sharded_device_index = 1
    rank = 19
    ret = _get_param_list_when_first_dim_sharded(device_arrange, first_dim_sharded_device_index, rank)

    assert ret == [16, 17, 18, 19, 20, 21, 22, 23]
