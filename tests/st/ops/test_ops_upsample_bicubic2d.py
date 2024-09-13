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
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore import context, ops


def set_mode(mode):
    if mode == "GRAPH_MODE_O0":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    elif mode == "GRAPH_MODE":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@test_utils.run_with_cell
def upsample_bicubic2d_forward_func(x, size=None, scale_factor=None, align_corners=False):
    return ops.function.nn_func.interpolate_ext(x, size, scale_factor, "bicubic", align_corners)


@test_utils.run_with_cell
def upsample_bicubic2d_backward_func(x, size=None, scale_factor=None, align_corners=False):
    return ms.grad(upsample_bicubic2d_forward_func, (0,))(x, size, scale_factor, align_corners)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["GRAPH_MODE_O0", "PYNATIVE_MODE"])
def test_upsample_bicubic_2d(mode):
    """
    Feature: test ops.
    Description: test op UpsampleBicubic2D.
    Expectation: success.
    """
    set_mode(mode)
    input_tensor = Tensor(
        np.array(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]
        ).astype(np.float32)
    )
    expected = np.array(
        [
            [
                [
                    [0.1000, 0.1406, 0.2000, 0.2594, 0.3000],
                    [0.1944, 0.2351, 0.2944, 0.3538, 0.3944],
                    [0.3056, 0.3462, 0.4056, 0.4649, 0.5056],
                    [0.4000, 0.4406, 0.5000, 0.5594, 0.6000],
                ],
                [
                    [0.7000, 0.7406, 0.8000, 0.8594, 0.9000],
                    [0.7944, 0.8351, 0.8944, 0.9538, 0.9944],
                    [0.9056, 0.9462, 1.0056, 1.0649, 1.1056],
                    [1.0000, 1.0406, 1.1000, 1.1594, 1.2000],
                ],
            ]
        ]
    ).astype(np.float32)
    out = upsample_bicubic2d_forward_func(input_tensor, (4, 5), None, True)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [
            [
                [[2.8125, 4.3750, 2.8125], [2.8125, 4.3750, 2.8125]],
                [[2.8125, 4.3750, 2.8125], [2.8125, 4.3750, 2.8125]],
            ]
        ]
    ).astype(np.float32)
    out = upsample_bicubic2d_backward_func(input_tensor, (4, 5), None, True)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_upsample_bicubic_2d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleBicubic2D and UpsampleBicubic2DGrad.
    Expectation: expect UpsampleBicubic2D and UpsampleBicubic2DGrad result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 10), dtype=ms.float32)
    TEST_OP(
        upsample_bicubic2d_forward_func,
        [
            [input_case1, (100, 200), None, True],
            [input_case2, (40, 80), None, False],
        ],
        'upsample_bicubic2d',
        disable_mode=["GRAPH_MODE"],
        disable_input_check=True
    )
