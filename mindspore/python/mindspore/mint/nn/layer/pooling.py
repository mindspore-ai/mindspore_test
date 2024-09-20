# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""normalization for mint"""
from __future__ import absolute_import
from __future__ import division

from mindspore import mint
from mindspore.nn.cell import Cell


class _AdaptiveAvgPoolNd(Cell):
    """Common base of AdaptiveAvgPoolNd"""

    def __init__(self, output_size) -> None:
        super(_AdaptiveAvgPoolNd, self).__init__()
        self.output_size = output_size

    def extend_repr(self):
        return 'output_size={}'.format(self.output_size)


class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    r"""
    Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output is of size :math:`L_{out}` , for any input size.
    The number of output features is equal to the number of input planes.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        output_size (int): the target output size :math:`L_{out}` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, L_{in})` or :math:`(C, L_{in})` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([[[2, 1, 2], [2, 3, 5]]]), mindspore.float16)
        >>> net = mint.nn.AdaptiveAvgPool1d(3)
        >>> output = net(input)
        >>> print(output)
        [[[2. 1. 2.]
          [2. 3. 5.]]]
    """

    def construct(self, input):
        return mint.nn.functional.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size :math:`H x W` , for any input size.
    The number of output features is equal to the number of input planes.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        output_size (Union(int, tuple[int])): the target output size of the image of the form :math:`H x W` .
            Can be a tuple :math:`(H, W)` or a single :math:`H` for square image :math:`H x H` .
            :math:`H` and :math:`W` can be either a ``int`` , or ``None`` which means the size will
            be the same as that of the input.

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, H, W)` or :math:`(C, H, W)` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([[[2, 1, 2], [2, 3, 5]]]), mindspore.float16)
        >>> net = mint.nn.AdaptiveAvgPool2d((2, 2))
        >>> output = net(input)
        >>> print(output)
        [[[1.5 1.5]
          [2.5 4. ]]]
    """

    def construct(self, input):
        return mint.nn.functional.adaptive_avg_pool2d(input, self.output_size)


__all__ = [
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool1d',
]
