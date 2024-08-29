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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore as ms
from mindspore import nn
import mindspore.context as context
from mindspore.ops.auto_generate import TopKRouter


class TopKRouterNet(nn.Cell):
    def __init__(self):
        super(TopKRouterNet, self).__init__()
        self.topkrouter = TopKRouter()

    def construct(self, x_data, capacity_data, expert_num_data, drop_type):
        return self.topkrouter(x_data, capacity_data, expert_num_data, drop_type)


class TopKRouterNet1(nn.Cell):
    def __init__(self):
        super(TopKRouterNet1, self).__init__()
        self.topkrouter = TopKRouter()

    def construct(self, x_data, capacity_data, expert_num_data):
        return self.topkrouter(x_data, capacity_data, expert_num_data)


x = ms.Tensor([[[0, 1], [1, 3],
                [3, 2], [2, 2],
                [2, 0], [0, 1],
                [1, 2], [2, 1],
                [1, 2], [2, 0]]], ms.int32)
capacity = 3
expert_num = 4

truth_dispatch_idx_s = np.array([[[1, 5, 6],
                                  [1, 2, 6],
                                  [3, 4, 4],
                                  [2, 3, 0]]]).astype(np.int32)

truth_combine_idx_s = np.array([[[1, 5], [6, 13],
                                 [14, 9], [10, 11],
                                 [8, 2], [3, 7],
                                 [4, 8], [8, 4],
                                 [4, 8], [8, 0]]]).astype(np.int32)

truth_dispatch_idx_k = np.array([[[1, 6, 5],
                                  [2, 7, 9],
                                  [4, 5, 8],
                                  [3, 2, 0]]]).astype(np.int32)

truth_combine_idx_k = np.array([[[1, 4], [5, 14],
                                 [13, 8], [9, 8],
                                 [10, 3], [2, 4],
                                 [6, 8], [11, 4],
                                 [7, 8], [8, 0]]]).astype(np.int32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_topkrouter_default_input(mode):
    """
    Feature: topkrouter test in ascend.
    Description: The input shape is static.
    Expectation: expect correct forward result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    ms_net = TopKRouterNet1()
    dispatch_idx, combine_idx = ms_net(x, capacity, expert_num)
    print(dispatch_idx, combine_idx)
    np.testing.assert_allclose(dispatch_idx.asnumpy(), truth_dispatch_idx_s)
    np.testing.assert_allclose(combine_idx.asnumpy(), truth_combine_idx_s)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_topkrouter_sdrop(mode):
    """
    Feature: topkrouter test in ascend.
    Description: The input shape is static.
    Expectation: expect correct forward result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    ms_net = TopKRouterNet()
    dispatch_idx, combine_idx = ms_net(x, capacity, expert_num, 0)
    print(dispatch_idx, combine_idx)
    np.testing.assert_allclose(dispatch_idx.asnumpy(), truth_dispatch_idx_s)
    np.testing.assert_allclose(combine_idx.asnumpy(), truth_combine_idx_s)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_topkrouter_dynamic_shape_sdrop(mode):
    """
    Feature: topkrouter test in ascend.
    Description: test case with capacity is dynamic.
    Expectation: expect correct forward result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    ms_net = TopKRouterNet()
    capacity_dyn = ms.mutable(3)
    dispatch_idx, combine_idx = ms_net(x, capacity_dyn, expert_num, 0)
    np.testing.assert_allclose(dispatch_idx.asnumpy(), truth_dispatch_idx_s)
    np.testing.assert_allclose(combine_idx.asnumpy(), truth_combine_idx_s)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_topkrouter_kdrop(mode):
    """
    Feature: topkrouter test in ascend.
    Description: The input shape is static.
    Expectation: expect correct forward result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    ms_net = TopKRouterNet()
    dispatch_idx, combine_idx = ms_net(x, capacity, expert_num, 1)
    np.testing.assert_allclose(dispatch_idx.asnumpy(), truth_dispatch_idx_k)
    np.testing.assert_allclose(combine_idx.asnumpy(), truth_combine_idx_k)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_topkrouter_dynamic_shape_kdrop(mode):
    """
    Feature: topkrouter test in ascend.
    Description: test case with capacity is dynamic.
    Expectation: expect correct forward result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    ms_net = TopKRouterNet()
    capacity_dyn = ms.mutable(3)
    dispatch_idx, combine_idx = ms_net(x, capacity_dyn, expert_num, 1)
    np.testing.assert_allclose(dispatch_idx.asnumpy(), truth_dispatch_idx_k)
    np.testing.assert_allclose(combine_idx.asnumpy(), truth_combine_idx_k)
