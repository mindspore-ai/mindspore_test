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
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark

roi_align_grad = ms.ops.auto_generate.RoiAlignGradExt()

def roi_align_grad_case(data_type=np.float32):
    boxes = Tensor(np.array([[0, 2.0, 2.0, 8.0, 8.0]], data_type))
    dy = Tensor(np.array([[[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]]], data_type))
    input_shape = (1, 1, 3, 3)
    output_size = (3, 3)
    spatial_scale, sampling_ratio = 0.25, 2
    aligned = False

    output = roi_align_grad(dy, boxes, input_shape, output_size, spatial_scale, sampling_ratio, aligned)
    expect = [
        [
            [
                [0.00625, 0.075, 0.06875],
                [0.04375, 0.525, 0.48125],
                [0.025, 0.3, 0.275],
            ]
        ]
    ]
    np.testing.assert_allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_roi_align_grad_float32(mode):
    """
    Feature: Test the operator RoiAlignGradExt
    Description:  Test in GRAPH and PYNATIVE mode using float32 inputs
    Expectation: Assert the result is equal to the expectation
    """
    ms.context.set_context(mode=mode)
    roi_align_grad_case(np.float32)
