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
# pylint: disable=unused-variable
import pytest
import re
import os
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import ops
from tests.st.utils import test_utils


@test_utils.run_with_cell
def range_forward_func(start, limit, delta):
    return ops.range(start, limit, delta, maxlen=10)

def grep(keyword, path):
    files = os.listdir(path)
    for file in files:
        filename = path + file
        with open(filename, 'r') as file:
            for line in file:
                if re.search(keyword, line):
                    return True
    return False

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@test_utils.run_test_with_On
def test_range_no_heter_copy():
    """
    Feature: heter no copy generalization.
    Description: no heter copy.
    Expectation: success
    """
    context.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path='./test_no_heter_generalization')
    context.set_context(jit_level='O0')
    start = ms.Tensor([0])
    limit = ms.Tensor([10])
    delta = ms.Tensor([2])
    output = range_forward_func(start, limit, delta)
    expect_output = np.array([0, 2, 4, 6, 8]).astype(np.int64)
    np.testing.assert_array_equal(output.asnumpy(), expect_output)
    keyword = "copy dest device target"
    path = "./test_no_heter_generalization/actor_set/"
    assert not grep(keyword, path), "heter no copy generalization failed."
