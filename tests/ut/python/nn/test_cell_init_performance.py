# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
""" test cell init performance"""
import time

import mindspore.nn as nn


class EmptyCell(nn.Cell):
    # pylint: disable=W0235
    def __init__(self):
        super(EmptyCell, self).__init__()

    def construct(self, x):
        return x


def test_cell_init_performance():
    """
    Feature: test cell init performance.
    Description: Verify the result of cell init performance
    Expectation: success
    """
    # warm up
    for _ in range(100):
        EmptyCell()

    start = time.time()
    for _ in range(10000):
        EmptyCell()
    end = time.time()
    cost = end - start
    print(f'For create cell for 10000 times, average time cost {cost * 100} microseconds')
    assert cost < 1.5
