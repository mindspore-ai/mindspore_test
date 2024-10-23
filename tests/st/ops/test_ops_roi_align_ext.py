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
from mindspore import ops, Tensor
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

def roi_align_forward_func(x, boxes, output_size, spatial_scale, sampling_ratio, aligned):
    return ops.roi_align(x, boxes, output_size, spatial_scale, sampling_ratio, aligned)

def roi_align_case(data_type=np.float32):
    features = Tensor(
        np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6],
                        [7, 8, 9, 10, 11, 12],
                        [13, 14, 15, 16, 17, 18],
                        [19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30],
                        [31, 32, 33, 34, 35, 36],
                    ]
                ]
            ],
            data_type
        )
    )

    # test case 1
    boxes = Tensor(np.array([[0, 2.0, 2.0, 22.0, 22.0]], data_type))
    output_size = (2, 2)
    spatial_scale, sampling_ratio = 0.25, 2
    aligned = True
    output = ops.roi_align(features, boxes, output_size, spatial_scale, sampling_ratio, aligned)
    expect = [[[[9.75, 12.25], [24.75, 27.25]]]]
    np.testing.assert_allclose(output.asnumpy(), expect)

    # test case 2
    boxes = Tensor(np.array([[0, 3.0, 3.0, 21.0, 21.0]], data_type))
    output_size = (3, 3)
    spatial_scale, sampling_ratio = 0.25, 2
    aligned = False
    output = ops.roi_align(features, boxes, output_size, spatial_scale, sampling_ratio, aligned)
    expect = [[[[11.5, 13.0, 14.5], [20.5, 22.0, 23.5], [29.5, 31.0, 32.5]]]]
    np.testing.assert_allclose(output.asnumpy(), expect)

    # test case 3
    boxes = Tensor(np.array([[0, 2.0, 2.0, 6.0, 6.0]], data_type))
    output_size = (2, 2)
    spatial_scale, sampling_ratio = 1.0, -1
    aligned = True
    output = ops.roi_align(features, boxes, output_size, spatial_scale, sampling_ratio, aligned)
    expect = [[[[18.5, 20.5], [30.5, 32.5]]]]
    np.testing.assert_allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_roi_align_float32(mode):
    """
    Feature: Test the operator roi_align
    Description:  Test in GRAPH and PYNATIVE mode using float32 inputs
    Expectation: Assert the result is equal to the expectation
    """
    ms.context.set_context(mode=mode)
    roi_align_case(np.float32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_roi_align_float32_dynamic_shape():
    """
    Feature: Test the operator roi_align with dynamic shape inputs
    Description:  Test in GRAPH and PYNATIVE mode using float32 dynamic shape inputs using TEST_OP
    Expectation: Assert the result is equal to the expectation
    """
    x1 = Tensor(
        np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6],
                        [7, 8, 9, 10, 11, 12],
                        [13, 14, 15, 16, 17, 18],
                        [19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30],
                        [31, 32, 33, 34, 35, 36],
                    ]
                ]
            ],
            np.float32
        )
    )
    x2 = Tensor(
        np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [7, 8, 9, 10],
                        [13, 14, 15, 16],
                        [19, 20, 21, 22],
                    ]
                ]
            ],
            np.float32
        )
    )

    boxes1 = Tensor(np.array([[0, 1.0, 2.0, 3.0, 4.0]], np.float32))
    boxes2 = Tensor(np.array([[0, 2.0, 2.0, 5.0, 5.0]], np.float32))
    output_size = (2, 2)
    spatial_scale, sampling_ratio = 0.25, 2
    aligned = True
    TEST_OP(roi_align_forward_func, [[x1, boxes1, output_size, spatial_scale, sampling_ratio, aligned],
                                     [x2, boxes2, output_size, spatial_scale, sampling_ratio, aligned]],
            'roi_align_ext', disable_input_check=True, disable_mode=['GRAPH_MODE'])
