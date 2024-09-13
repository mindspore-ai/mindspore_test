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
"""activation layer for mint"""
from __future__ import absolute_import
from __future__ import division

from mindspore import mint
from mindspore.nn.cell import Cell


class SiLU(Cell):
    r"""
    Applies the silu linear unit function element-wise.

    .. math::

        \text{SiLU}(x) = x * \sigma(x),

    where :math:`x_i` is an element of the input, :math:`\sigma(x)` is Sigmoid function.

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    SiLU Activation Function Graph:

    .. image:: ../images/SiLU.png
        :align: center

    Inputs:
        - **input** (Tensor) - `input` is :math:`x` in the preceding formula.
          Input with the data type float16 or float32. Tensor of any dimension.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If dtype of `input` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> silu = mint.nn.SiLU()
        >>> output = silu(input)
        >>> print(output)
        [-0.269  1.762  -0.1423  1.762  -0.269]
    """

    def __init__(self):
        """Initialize SiLU."""
        super(SiLU, self).__init__()

    def construct(self, x):
        return mint.nn.functional.silu(x)


__all__ = [
    'SiLU',
]
