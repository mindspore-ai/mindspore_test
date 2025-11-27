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
"""test distributed tensor"""

import numpy as np
from mindspore.communication.management import init
from mindspore import Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore.parallel import Layout
from tests.st.auto_parallel.utils import distribute_tensor


def _prepare_input_data():
    world_size = get_group_size()
    np_data = np.arange(world_size ** 2).reshape(world_size, world_size)
    data = Tensor(np_data)
    if not data.is_contiguous():
        data = data.contiguous()
    return data

def test_distribute_on_1dmesh():
    """
    Feature: dtensor test
    Description: test_distribute_on_1dmesh
    Expectation: run success
    """
    init()
    rank_id = get_rank()
    world_size = get_group_size()
    global_tensor = _prepare_input_data()
    rank_list = list(range(world_size))
    base_layout = Layout(device_matrix=(8,), alias_name=("dp",), rank_list=rank_list)
    # shard on dim of "dp"
    dp_shard_layout = base_layout("dp")
    local_slice = distribute_tensor(global_tensor, dp_shard_layout, src_data_rank=0)
    mask = local_slice.asnumpy() == global_tensor.asnumpy()[rank_id:rank_id + 1]
    assert np.all(mask)

    # replicate on dim of "dp"
    replicate_layout = base_layout("None")
    local_slice = distribute_tensor(global_tensor, replicate_layout, src_data_rank=0)
    mask = local_slice.asnumpy() == global_tensor.asnumpy()
    assert np.all(mask)

def test_distribute_on_2dmesh():
    """
    Feature: dtensor test
    Description: test_distribute_on_2dmesh
    Expectation: run success
    """
    init()
    rank_id = get_rank()
    world_size = get_group_size()
    global_tensor = _prepare_input_data()
    rank_list = list(range(world_size))
    layout0 = Layout(device_matrix=(2, 4), alias_name=("replicate", "dp"), rank_list=rank_list)
    hsdp_shard_layout = layout0("dp", "None")
    local_slice = distribute_tensor(global_tensor, hsdp_shard_layout, src_data_rank=0)
    assert local_slice.layout and local_slice.shape == (world_size, world_size)
    local_slice_dim0_expect_size = global_tensor.shape[0] // 4
    if rank_id in [0, 4]:
        start = 0
        end = local_slice_dim0_expect_size
        mask = local_slice.asnumpy() == global_tensor.asnumpy()[start:end]
        assert np.all(mask)
    elif rank_id in [1, 5]:
        start = local_slice_dim0_expect_size
        end = 2 * local_slice_dim0_expect_size
        mask = local_slice.asnumpy() == global_tensor.asnumpy()[start:end]
        assert np.all(mask)
    elif rank_id in [2, 6]:
        start = 2 *local_slice_dim0_expect_size
        end = 3 * local_slice_dim0_expect_size
        mask = local_slice.asnumpy() == global_tensor.asnumpy()[start:end]
        assert np.all(mask)
    elif rank_id in [3, 7]:
        start = 3 *local_slice_dim0_expect_size
        end = 4 * local_slice_dim0_expect_size
        mask = local_slice.asnumpy() == global_tensor.asnumpy()[start:end]
        assert np.all(mask)
