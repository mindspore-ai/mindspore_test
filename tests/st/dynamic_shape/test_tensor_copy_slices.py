# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.ops import auto_generate as ops


class TensorCopySlicesNet(nn.Cell):
    def __init__(self):
        super(TensorCopySlicesNet, self).__init__()
        self.tensor_copy_slices = ops.TensorCopySlices()

    def construct(self, x, value, begin, end, strides):
        return self.tensor_copy_slices(x, value, begin, end, strides)


def test_tensor_copy_slices():
    """
    Feature: DynamicShape.
    Description: Test TensorCopySlices.
    Expectation: No exception.
    """
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
    reserved_env = os.getenv('MS_DEV_PRECOMPILE_ONLY')
    os.environ['MS_DEV_PRECOMPILE_ONLY'] = '1'
    net = TensorCopySlicesNet()
    out = net(ms.Tensor(np.zeros((5, 5))), ms.Tensor(np.ones((2, 5))), (3, 0), (5, 5), (1, 1))
    if reserved_env is None:
        os.unsetenv('MS_DEV_PRECOMPILE_ONLY')
    else:
        os.environ['MS_DEV_PRECOMPILE_ONLY'] = reserved_env
