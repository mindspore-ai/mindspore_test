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


from mindspore.parallel._tensor import _get_resharding_operator_map, _get_pipeline_operator_map


def test_merge_pipeline_stages_2_1():
    """
    Feature: generate op list for merging pipeline stages
    Description: pp2 -> pp1
    Expectation: assert no error.
    """
    from_layout = ([2, 2], [1, 0], [8, 8], [0, 1, 2, 3])
    to_layout = ([4, 2], [1, 0], [8, 8], [0, 1, 2, 3, 4, 5, 6, 7])
    for i in range(4):
        pp_op_map = _get_pipeline_operator_map(from_layout, to_layout, i)
        broadcast_ground_truth = {i % 4: [('Broadcast', i % 4, [i % 4, (i % 4) + 4])],
                                  (i % 4) + 4: [('Broadcast', i % 4, [i % 4, (i % 4) + 4])]}
        assert pp_op_map == broadcast_ground_truth, \
            f"rank id is {i}, Broadcast args is not expected. expect {broadcast_ground_truth}, but got {pp_op_map}"


def test_merge_pipeline_stages_2_2():
    """
    Feature: generate op list for merging pipeline stages
    Description: pp2 -> pp2
    Expectation: assert no error.
    """
    from_layout = ([2, 2], [1, 0], [8, 8], [0, 1, 2, 3])
    to_layout = ([4, 1], [1, 0], [8, 8], [0, 1, 2, 3])
    pp_op_map = _get_pipeline_operator_map(from_layout, to_layout, 0)
    assert pp_op_map == {}, \
        f"rank id is 0, Broadcast args is not expected. expect {{}}, but got {pp_op_map}"


def test_merge_pipeline_stages_2_4():
    """
    Feature: generate op list for merging pipeline stages
    Description: pp2 -> pp4
    Expectation: assert no error.
    """
    from_layout = ([2, 2], [1, 0], [8, 8], [0, 1, 2, 3])
    to_layout = ([4, 1], [1, 0], [8, 8], [0, 1])
    pp_op_map = _get_pipeline_operator_map(from_layout, to_layout, 0)
    assert pp_op_map == {}, \
        f"rank id is 0, Broadcast args is not expected. expect {{}}, but got {pp_op_map}"


def test_merge_pipeline_stages_4_2():
    """
    Feature: generate op list for merging pipeline stages
    Description: pp4 -> pp2
    Expectation: assert no error.
    """
    from_layout = ([2, 2], [1, 0], [8, 8], [0, 1])
    to_layout = ([4, 1], [1, 0], [8, 8], [0, 1, 2, 3])
    for i in range(4):
        pp_op_map = _get_pipeline_operator_map(from_layout, to_layout, i)
        broadcast_ground_truth = {i % 2: [('Broadcast', i % 2, [i % 2, (i % 2) + 2])],
                                  (i % 2) + 2: [('Broadcast', i % 2, [i % 2, (i % 2) + 2])]}
        assert pp_op_map == broadcast_ground_truth, \
            f"rank id is {i}, Broadcast args is not expected. expect {broadcast_ground_truth}, but got {pp_op_map}"


def test_reshard_operator_map_1():
    """
    Feature: generate mp op map
    Description: mp4 -> mp2
    Expectation: assert no error.
    """
    rank_list = list(range(8))
    from_layout = ([2, 4], [0, -1], [8, 8], rank_list)
    to_layout = ([2, 2, 2], [1, -1], [8, 8], rank_list)
    reshard_op_map = _get_resharding_operator_map(from_layout, to_layout, 0)
    reshard_ground_truth = {0: [('AllConcat', [0, 1, 0])], 1: [('AllConcat', [0, 1, 0])]}
    assert reshard_op_map == reshard_ground_truth, \
        f"rank id is 0, reshard map is not expected. expect {reshard_ground_truth}, but got {reshard_op_map}"


def test_reshard_operator_map_2():
    """
    Feature: generate mp op map
    Description: mp4 -> mp2
    Expectation: assert no error.
    """
    rank_list = list(range(8))
    from_layout = ([4, 2], [1, -1], [8, 8], rank_list)
    to_layout = ([2, 2, 2], [1, -1], [8, 8], rank_list)
    reshard_op_map = _get_resharding_operator_map(from_layout, to_layout, 0)
    reshard_ground_truth = {
        2: [('AllConcat', [0, 2, 0]), ('Reshape', [1, 4, 8]), ('AllConcat', [2, 6, 0]), ('Reshape', [8, 8]),
            ('StridedSlice', [4, 0, 8, 8, 1, 1])],
        6: [('AllConcat', [4, 6, 0]), ('Reshape', [1, 4, 8]), ('AllConcat', [2, 6, 0]), ('Reshape', [8, 8]),
            ('StridedSlice', [4, 0, 8, 8, 1, 1])],
        0: [('AllConcat', [0, 2, 0]), ('Reshape', [1, 4, 8]), ('AllConcat', [0, 4, 0]), ('Reshape', [8, 8]),
            ('StridedSlice', [0, 0, 4, 8, 1, 1])],
        4: [('AllConcat', [4, 6, 0]), ('Reshape', [1, 4, 8]), ('AllConcat', [0, 4, 0]), ('Reshape', [8, 8]),
            ('StridedSlice', [0, 0, 4, 8, 1, 1])]}
    assert reshard_op_map == reshard_ground_truth, \
        f"rank id is 0, reshard map is not expected. expect {reshard_ground_truth}, but got {reshard_op_map}"


def test_reshard_operator_map_3():
    """
    Feature: generate mp op map
    Description: mp2 + op -> mp2
    Expectation: assert no error.
    """
    rank_list = list(range(8))
    from_layout = ([2, 2, 2], [[0, 2, 1], -1], [8, 8], rank_list)
    to_layout = ([4, 2], [0, -1], [8, 8], rank_list)
    reshard_op_map = _get_resharding_operator_map(from_layout, to_layout, 0)
    reshard_ground_truth = {2: [('Reshape', [1, 1, 1, 8]), ('Reshape', [1, 8]), ('AllConcat', [0, 2, 4, 6, 0])],
                            0: [('Reshape', [1, 1, 1, 8]), ('Reshape', [1, 8]), ('AllConcat', [0, 2, 4, 6, 0])],
                            4: [('Reshape', [1, 1, 1, 8]), ('Reshape', [1, 8]), ('AllConcat', [0, 2, 4, 6, 0])],
                            6: [('Reshape', [1, 1, 1, 8]), ('Reshape', [1, 8]), ('AllConcat', [0, 2, 4, 6, 0])]}
    assert reshard_op_map == reshard_ground_truth, \
        f"rank id is 0, reshard map is not expected. expect {reshard_ground_truth}, but got {reshard_op_map}"


def test_reshard_operator_map_4():
    """
    Feature: generate mp op map
    Description: mp2 + op2 -> mp2
    Expectation: assert no error.
    """
    rank_list = list(range(8))
    from_layout = ([2, 2, 2], [[0, 1], -1], [8, 8], rank_list)
    to_layout = ([4, 2], [0, -1], [8, 8], rank_list)
    reshard_op_map = _get_resharding_operator_map(from_layout, to_layout, 0)
    reshard_ground_truth = {2: [('Reshape', [1, 2, 8]), ('Reshape', [2, 8]), ('AllConcat', [0, 2, 0])],
                            0: [('Reshape', [1, 2, 8]), ('Reshape', [2, 8]), ('AllConcat', [0, 2, 0])]}
    assert reshard_op_map == reshard_ground_truth, \
        f"rank id is 0, reshard map is not expected. expect {reshard_ground_truth}, but got {reshard_op_map}"


def test_reshard_operator_map_5():
    """
    Feature: generate mp op map
    Description: mp2ep2 + op -> mp2
    Expectation: assert no error.
    """
    rank_list = list(range(8))
    from_layout = ([2, 2, 2], [[0, 2], 1], [8, 8], rank_list)
    to_layout = ([4, 2], [0, -1], [8, 8], rank_list)
    reshard_op_map = _get_resharding_operator_map(from_layout, to_layout, 0)
    reshard_ground_truth = {
        2: [('Reshape', [1, 2, 4]), ('AllConcat', [0, 2, 2]), ('AllConcat', [2, 6, 0]), ('Reshape', [4, 8])],
        6: [('Reshape', [1, 2, 4]), ('AllConcat', [4, 6, 2]), ('AllConcat', [2, 6, 0]), ('Reshape', [4, 8])],
        0: [('Reshape', [1, 2, 4]), ('AllConcat', [0, 2, 2]), ('AllConcat', [0, 4, 0]), ('Reshape', [4, 8])],
        4: [('Reshape', [1, 2, 4]), ('AllConcat', [4, 6, 2]), ('AllConcat', [0, 4, 0]), ('Reshape', [4, 8])]}
    assert reshard_op_map == reshard_ground_truth, \
        f"rank id is 0, reshard map is not expected. expect {reshard_ground_truth}, but got {reshard_op_map}"
