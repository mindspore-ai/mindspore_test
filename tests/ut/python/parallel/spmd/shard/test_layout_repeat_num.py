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
from mindspore.parallel.shard import Layout
from parallel.spmd.shard._utils import init_hccl

def test_layout_repeat_num():
    '''
    Feature: repeat num.
    Description: Test repeat num for layout.
    Expectation: Run success.
    '''
    init_hccl(global_rank=0, device_num=2 * 4 * 6 * 8)

    layout = Layout((2, 4, 6, 8), ("a", "b", "c", "d"))
    with pytest.raises(ValueError):
        layout.repeat_num()

    x_layout = layout("a", "c")
    y_layout = layout("b", "None")
    z_layout = layout(("a", "b"), "c", "None")

    x_num = x_layout.repeat_num()
    y_num = y_layout.repeat_num()
    z_num = z_layout.repeat_num()

    assert x_num == 32
    assert y_num == 96
    assert z_num == 8


def test_layout_repeat_num_for_pp_last_stage():
    '''
    Feature: repeat num.
    Description: Test repeat num for last stage.
    Expectation: Run success.
    '''
    init_hccl(global_rank=6, device_num=8)

    layout = Layout((2, 2), ("dp", "mp"), rank_list=[4, 5, 6, 7])
    x_layout = layout("dp", "None")
    x_num = x_layout.repeat_num()
    assert x_num == 2


def test_layout_repeat_num_for_pp_first_stage():
    '''
    Feature: repeat num.
    Description: Test repeat num for first stage.
    Expectation: Run success.
    '''
    init_hccl(global_rank=1, device_num=8)

    layout = Layout((2, 2), ("dp", "mp"), rank_list=[0, 1, 2, 3])
    x_layout = layout("dp", "None")
    x_num = x_layout.repeat_num()
    assert x_num == -1
