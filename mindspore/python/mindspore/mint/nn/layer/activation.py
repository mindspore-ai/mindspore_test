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
    Calculates the SiLU activation function element-wise. It is also sometimes referred to as Swish
    function.

    The SiLU function is defined as follows:

    .. math::

        \text{SiLU}(x) = x * \sigma(x),

    where :math:`x_i` is an element of the input, :math:`\sigma(x)` is Sigmoid function.

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    SiLU Activation Function Graph:

    .. image:: ../images/SiLU.png
        :align: center

    .. warning::
        This is an experimental API that is subject to change or deletion.

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


class LogSigmoid(Cell):
    r"""
    Applies logsigmoid activation element-wise. The input is a Tensor with any valid shape.

    Logsigmoid is defined as:

    .. math::
        \text{logsigmoid}(x_{i}) = \log(\frac{1}{1 + \exp(-x_i)}),

    where :math:`x_{i}` is the element of the input.

    LogSigmoid Activation Function Graph:

    .. image:: ../images/LogSigmoid.png
        :align: center

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **input** (Tensor) - The input of LogSigmoid with data type of bfloat16, float16 or float32.
          The shape is :math:`(*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If dtype of `input` is not bfloat16, float16 and float32.
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> net = mint.nn.LogSigmoid()
        >>> input = Tensor([1.0, 2.0, 3.0], mindspore.float32)
        >>> output = net(input)
        >>> print(output)
        [-0.31326166 -0.12692806 -0.04858734]
    """
    def __init__(self):
        """Initialize LogSigmoid."""
        super(LogSigmoid, self).__init__()

    def construct(self, input):
        return mint.nn.functional.logsigmoid(input)


class ELU(Cell):
    r"""
    Exponential Linear Unit activation function

    Applies the exponential linear unit function element-wise.The activation function is defined as:

    .. math::
        ELU_{i} =
        \begin{cases}
        x_i, &\text{if } x_i \geq 0; \cr
        \alpha * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    where :math:`x_i` represents the element of the input and :math:`\alpha` represents the `alpha` parameter.

    ELU Activation Function Graph:

    .. image:: ../images/ELU.png
        :align: center

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        alpha (float, optional): The alpha value of ELU, the data type is float. Default: ``1.0`` .

    Inputs:
        - **input** (Tensor) - The input of ELU is a Tensor of any dimension.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If `alpha` is not a float.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
        >>> elu = mint.nn.ELU()
        >>> result = elu(input)
        >>> print(result)
        [-0.63212055  -0.86466473  0.  2.  1.]
    """

    def __init__(self, alpha=1.0):
        """Initialize ELU."""
        super(ELU, self).__init__()
        self.alpha = alpha

    def construct(self, input):
        return mint.nn.functional.elu(input, self.alpha)


class Tanh(Cell):
    r"""
    Applies the Tanh function element-wise, returns a new tensor with the hyperbolic tangent of the elements of input.

    Tanh function is defined as:

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Tanh Activation Function Graph:

    .. image:: ../images/Tanh.png
        :align: center

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **input** (Tensor) - Tensor of any dimension, input with data type of float16 or float32.

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
        >>> input = Tensor(np.array([1, 2, 3, 2, 1]), mindspore.float16)
        >>> tanh = mint.nn.Tanh()
        >>> output = tanh(input)
        >>> print(output)
        [0.7617 0.964  0.995  0.964  0.7617]
    """

    def __init__(self):
        """Initialize Tanh."""
        super(Tanh, self).__init__()

    def construct(self, input):
        return mint.nn.functional.tanh(input)


__all__ = [
    'LogSigmoid',
    'SiLU',
    'ELU',
    'Tanh',
]
