# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""activation"""
from __future__ import absolute_import

import numpy as np

from mindspore import _checkparam as validator
from mindspore._extends import cell_attr_register
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore.ops.operations import nn_ops as NN_OPS
from mindspore.nn.cell import Cell
from mindspore.ops.primitive import _primexpr

__all__ = ['Softmin',
           'Softmax',
           'Softmax2d',
           'LogSoftmax',
           'LogSoftmaxExt',
           'ReLU',
           'ReLU6',
           'RReLU',
           'SeLU',
           'SiLU',
           'Tanh',
           'Tanhshrink',
           'Hardtanh',
           'GELU',
           'FastGelu',
           'Sigmoid',
           'Softsign',
           'PReLU',
           'PReLUExt',
           'get_activation',
           'LeakyReLU',
           'HSigmoid',
           'HSwish',
           'ELU',
           'LogSigmoid',
           'LRN',
           'SoftShrink',
           'HShrink',
           'CELU',
           'Threshold',
           'Mish',
           'GLU'
           ]


class CELU(Cell):
    r"""
    CELU Activation Operator.

    Applies the continuously differentiable exponential linear units function element-wise.

    .. math::

        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    For more details, refer to `CELU <https://arxiv.org/abs/1704.07483>`_ .

    CELU Activation Function Graph:

    .. image:: ../images/CELU.png
        :align: center

    Args:
        alpha (float, optional): The :math:`\alpha` value for the Celu formulation. Default: ``1.0`` .

    Inputs:
        - **x** (Tensor) - The input of CELU. The required dtype is float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `alpha` is not a float.
        ValueError: If `alpha` has the value of 0.
        TypeError: If `x` is not a Tensor.
        TypeError: If the dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([-2.0, -1.0, 1.0, 2.0]), mindspore.float32)
        >>> celu = nn.CELU()
        >>> output = celu(x)
        >>> print(output)
        [-0.86466473 -0.63212055  1.          2.        ]
    """

    def __init__(self, alpha=1.0):
        """Initialize CELU."""
        super(CELU, self).__init__()
        self.celu = ops.CeLU(alpha=alpha)

    def construct(self, x):
        return self.celu(x)


class Softmin(Cell):
    r"""
    Softmin activation function, which is a two-category function :class:`mindspore.nn.Sigmoid` in the promotion of
    multi-classification, and the purpose is to show the results of multi-classification in the form of probability.

    Calculate the value of the exponential function for the elements of the input Tensor on the `axis`, and then
    normalized to lie in range [0, 1] and sum up to 1.

    Softmin is defined as:

    .. math::
        \text{softmin}(x_{i}) =  \frac{\exp(-x_i)}{\sum_{j=0}^{n-1}\exp(-x_j)},

    where :math:`x_{i}` is the :math:`i`-th slice in the given dimension of the input Tensor.

    Args:
        axis (Union[int, tuple[int]], optional): The axis to apply Softmin operation,
            if the dimension of input `x` is x.ndim,
            the range of axis is :math:`[-x.ndim, x.ndim)`. -1 means the last dimension.
            Default: ``-1`` . In CPU environment, `axis` only supports int type.

    Inputs:
        - **x** (Tensor) - Tensor for computing Softmin functions with data type of float16 or float32.

    Outputs:
        Tensor, which has the same type and shape as `x` with values in the range :math:`[0, 1]`.

    Raises:
        TypeError: If `axis` is neither an int nor a tuple.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `axis` is a tuple whose length is less than 1.
        ValueError: If `axis` is a tuple whose elements are not all in the range :math:`[-x.ndim, x.ndim)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> # axis = -1(default), and the sum of return value is 1.0.
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> softmin = nn.Softmin()
        >>> output = softmin(x)
        >>> print(output)
        [0.2341  0.636  0.0862  0.01165  0.03168 ]
    """

    def __init__(self, axis=-1):
        """Initialize Softmin."""
        super(Softmin, self).__init__()
        self.axis = axis

    def construct(self, x):
        return ops.function.softmin(x, self.axis)


class Softmax2d(Cell):
    r"""
    Softmax function applied to 2D features data.

    Applies `Softmax` to each location with an input Tensor of shape :math:`(C, H, W)` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`.
          The input of Softmax with data type of float16 or float32.

    Outputs:
        Tensor, which has the same type and shape as `x` with values in the range[0,1].

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `data_format` is neither 'NCHW' nor 'CHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[[[0.1, 0.2]], [[0.3, 0.4]], [[0.6, 0.5]]]]), mindspore.float32)
        >>> softmax2d = nn.Softmax2d()
        >>> output = softmax2d(x)
        >>> print(output)
        [[[[0.25838965 0.28001308]]
        [[0.31559783 0.34200877]]
        [[0.42601252 0.37797815]]]]
    """

    def __init__(self):
        """Initialize Softmax2d."""
        super(Softmax2d, self).__init__()
        self.softmax = ops.Softmax(axis=-3)
        self.shape = ops.Shape()

    @staticmethod
    @_primexpr
    def _check_input_dim(shape, cls_name):
        dim = len(shape)
        if dim not in (3, 4):
            raise ValueError(f"For '{cls_name}', the in_shape must have 3 or 4 dims, but got {dim}.")

    def construct(self, x):
        x_shape = self.shape(x)
        self._check_input_dim(x_shape, self.cls_name)
        return self.softmax(x)


class Softmax(Cell):
    r"""
    Softmax activation function, which is a two-category function :class:`mindspore.nn.Sigmoid` in the promotion of
    multi-classification, the purpose is to show the results of multi-classification in the form of probability.

    Calculate the value of the exponential function for the elements of the input Tensor on the `axis`, and then
    normalized to lie in range [0, 1] and sum up to 1.

    Softmax is defined as:

    .. math::
        \text{softmax}(input_{i}) =  \frac{\exp(input_i)}{\sum_{j=0}^{n-1}\exp(input_j)},

    where :math:`input_{i}` is the :math:`i`-th slice in the given dimension of the input Tensor.

    Args:
        axis (int, optional): The axis to apply Softmax operation, if the dimension of `input` is input.ndim,
            the range of axis is `[-input.ndim, input.ndim)`, -1 means the last dimension. Default: ``-1`` .

    Inputs:
        - **input** (Tensor) - The input of Softmax.

    Outputs:
        Tensor, which has the same type and shape as `input` with values in the range[0, 1].

    Raises:
        TypeError: If `axis` is neither an int nor a tuple.
        ValueError: If `axis` is a tuple whose length is less than 1.
        ValueError: If `axis` is a tuple whose elements are not all in range `[-input.ndim, input.ndim)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> # axis = -1(default), and the sum of return value is 1.0.
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> softmax = nn.Softmax()
        >>> output = softmax(input)
        >>> print(output)
        [0.03168 0.01166 0.0861  0.636   0.2341 ]
    """

    def __init__(self, axis=-1):
        """Initialize Softmax."""
        super(Softmax, self).__init__()
        self.softmax = ops.Softmax(axis)

    def construct(self, input):
        return self.softmax(input)


class SoftmaxExt(Cell):
    r"""
    Applies the Softmax function to an n-dimensional input Tensor.

    For details, please refer to :func:`mindspore.mint.nn.functional.softmax`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> softmax = nn.SoftmaxExt()
        >>> output = softmax(input)
        >>> print(output)
        [0.03168 0.01166 0.0861  0.636   0.2341 ]
    """

    def __init__(self, dim=None):
        """Initialize Softmax."""
        super(SoftmaxExt, self).__init__()
        self.dim = dim

    def construct(self, input):
        return ops.function.nn_func.softmax_ext(input, self.dim)


class LogSoftmax(Cell):
    r"""
    Applies the LogSoftmax function to n-dimensional input tensor element-wise.

    The input is transformed by the Softmax function and then by the log function to lie in range[-inf,0).

    Logsoftmax is defined as:

    .. math::

        \text{logsoftmax}(x_i) = \log \left(\frac{\exp(x_i)}{\sum_{j=0}^{n-1} \exp(x_j)}\right)

    Args:
        axis (int): The axis to apply LogSoftmax operation, -1 means the last dimension. Default: ``-1`` .

    Inputs:
        - **x** (Tensor) - The input of LogSoftmax, with float16 or float32 data type.

    Outputs:
        Tensor, which has the same type and shape as `x` with output values in the range[-inf,0).

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `axis` is not in range [-len(x), len(x)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> log_softmax = nn.LogSoftmax()
        >>> output = log_softmax(x)
        >>> print(output)
        [[-5.00672150e+00 -6.72150636e-03 -1.20067215e+01]
         [-7.00091219e+00 -1.40009127e+01 -9.12250078e-04]]
    """

    def __init__(self, axis=-1):
        """Initialize LogSoftmax."""
        super(LogSoftmax, self).__init__()
        self.log_softmax = ops.LogSoftmax(axis)

    def construct(self, x):
        return self.log_softmax(x)


class LogSoftmaxExt(Cell):
    r"""
    Applies the Log Softmax function to the input tensor on the specified axis.
    Supposes a slice in the given axis, :math:`x` for each element :math:`x_i`,
    the Log Softmax function is shown as follows:

    .. math::
        \text{output}(x_i) = \log \left(\frac{\exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    where :math:`N` is the length of the Tensor.

    Args:
        dim (int, optional): The axis to perform the Log softmax operation. Default: ``None`` .

    Returns:
        Tensor, with the same shape as the input.

    Raises:
        ValueError: If `dim` is not in range [-len(input.shape), len(input.shape)).

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> log_softmax = nn.LogSoftmaxExt(dim=-1)
        >>> output = log_softmax(x)
        >>> print(output)
        [[-5.00672150e+00 -6.72150636e-03 -1.20067215e+01]
         [-7.00091219e+00 -1.40009127e+01 -9.12250078e-04]]
    """

    def __init__(self, dim=None):
        """Initialize LogSoftmaxExt."""
        super(LogSoftmaxExt, self).__init__()
        self.log_softmax = ops.LogSoftmaxExt()
        self.dim = dim

    def construct(self, x):
        return self.log_softmax(x, dim=self.dim)


class ELU(Cell):
    r"""
    Applies the exponential linear unit function element-wise.

    The activation function is defined as:

    .. math::
        E_{i} =
        \begin{cases}
        x_i, &\text{if } x_i \geq 0; \cr
        \alpha * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    where :math:`x_i` represents the element of the input and :math:`\alpha` represents the `alpha` parameter.

    ELU Activation Function Graph:

    .. image:: ../images/ELU.png
        :align: center

    Args:
        alpha (float): The alpha value of ELU, the data type is float. Default: ``1.0`` .
            Only alpha equal to ``1.0`` is supported currently.

    Inputs:
        - **input_x** (Tensor) - The input of ELU is a Tensor of any dimension with data type of float16 or float32.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If `alpha` is not a float.
        TypeError: If dtype of `input_x` is neither float16 nor float32.
        ValueError: If `alpha` is not equal to 1.0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
        >>> elu = nn.ELU()
        >>> result = elu(x)
        >>> print(result)
        [-0.63212055  -0.86466473  0.  2.  1.]
    """

    def __init__(self, alpha=1.0):
        """Initialize ELU."""
        super(ELU, self).__init__()
        self.elu = ops.Elu(alpha)

    def construct(self, x):
        return self.elu(x)


class ReLU(Cell):
    r"""
    Applies ReLU (Rectified Linear Unit activation function) element-wise.

    .. math::

        \text{ReLU}(input) = (input)^+ = \max(0, input),

    It returns element-wise :math:`\max(0, input)`.

    .. note::
        The neurons with the negative output
        will be suppressed and the active neurons will stay the same.

    ReLU Activation Function Graph:

    .. image:: ../images/ReLU.png
        :align: center

    Inputs:
        - **input** (Tensor) - The input of ReLU is a Tensor of any dimension.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If dtype of `input` is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> relu = nn.ReLU()
        >>> output = relu(input)
        >>> print(output)
        [0. 2. 0. 2. 0.]
    """

    def __init__(self):
        """Initialize ReLU."""
        super(ReLU, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, input):
        return self.relu(input)


class ReLU6(Cell):
    r"""
    Compute ReLU6 activation function element-wise.

    ReLU6 is similar to ReLU with a upper limit of 6, which if the inputs are greater than 6, the outputs
    will be suppressed to 6.
    It computes element-wise as

    .. math::

        Y = \min(\max(0, x), 6)

    ReLU6 Activation Function Graph:

    .. image:: ../images/ReLU6.png
        :align: center

    Inputs:
        - **x** (Tensor) - The input of ReLU6 with data type of float16 or float32 and that
          is a Tensor of any valid shape.

    Outputs:
        Tensor, which has the same type as `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> relu6 = nn.ReLU6()
        >>> output = relu6(x)
        >>> print(output)
        [0. 0. 0. 2. 1.]
    """

    def __init__(self):
        """Initialize ReLU6."""
        super(ReLU6, self).__init__()
        self.relu6 = ops.ReLU6()

    def construct(self, x):
        return self.relu6(x)


class LeakyReLU(Cell):
    r"""
    Leaky ReLU activation function.

    The activation function is defined as:

    .. math::
            \text{leaky_relu}(x) = \begin{cases}x, &\text{if } x \geq 0; \cr
            {\alpha} * x, &\text{otherwise.}\end{cases}

    where :math:`\alpha` represents the `alpha` parameter.

    For more details, see `Rectifier Nonlinearities Improve Neural Network Acoustic Models
    <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_.

    LeakyReLU Activation Function Graph:

    .. image:: ../images/LeakyReLU.png
        :align: center

    Args:
        alpha (Union[int, float]): Slope of the activation function at x < 0. Default: ``0.2`` .

    Inputs:
        - **x** (Tensor) - The input of LeakyReLU is a Tensor of any dimension.

    Outputs:
        Tensor, has the same type and shape as the `x`.

    Raises:
        TypeError: If `alpha` is not a float or an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> leaky_relu = nn.LeakyReLU()
        >>> output = leaky_relu(x)
        >>> print(output)
        [[-0.2  4.  -1.6]
         [ 2.  -1.   9. ]]
    """

    def __init__(self, alpha=0.2):
        """Initialize LeakyReLU."""
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def construct(self, x):
        out = ops.leaky_relu(x, self.alpha)
        return out


class RReLU(Cell):
    r"""
    Applies RReLU (Randomized Leaky ReLU activation function) element-wise.

    The activation function is defined as:

    .. math::
        \text{RReLU}(x_{ji}) = \begin{cases}x_{ji}, &\text{if } x_{ji} \geq 0; \cr
        {\alpha_{ji}} * x_{ji}, &\text{otherwise.}\end{cases}

    where :math:`\alpha_{ji}` ~ :math:`U(l, u)`, :math:`l \le u`.

    Applies the RReLU function elementally, as described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolution Network <https://arxiv.org/pdf/1505.00853.pdf>`_ .

    Args:
        lower (Union[int, float]): Slope of the activation function at x < 0. Default: ``1 / 8`` .
        upper (Union[int, float]): Slope of the activation function at x < 0. Default: ``1 / 3`` .

    Inputs:
        - **x** (Tensor) - The input of RReLU is a Tensor of any dimension.

    Outputs:
        Tensor, after RReLU, has the same type and shape as the `x`.

    Raises:
        TypeError: If `lower` is not a float or an int.
        TypeError: If `upper` is not a float or an int.
        TypeError: If `x` is not a Tensor.
        TypeError: If `x` is not a Tensor of mindspore.float16 or mindspore.float32.
        ValueError: If `lower` is greater than upper.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0], [2.0, 0]]), mindspore.float32)
        >>> r_relu = nn.RReLU()
        >>> output = r_relu(x)
        >>> print(output)
        [[-0.31465699  4.        ]
         [ 2.          0.        ]]
    """

    def __init__(self, lower=1 / 8, upper=1 / 3):
        super(RReLU, self).__init__()
        validator.check_value_type('upper', upper, [float, int], self.cls_name)
        validator.check_value_type('lower', lower, [float, int], self.cls_name)
        if lower > upper:
            raise ValueError(f"For {self.cls_name}, the value of 'upper' must be greater than or equal to 'lower', "
                             f"but got upper: {upper}, lower: {lower}. ")
        self.lower = Tensor(lower, dtype=mstype.float32)
        self.upper = Tensor(upper, dtype=mstype.float32)
        self.sign = ops.Sign()

    def construct(self, x):
        if not isinstance(x, Tensor):
            raise TypeError(f"For 'rrelu', the input must be a Tensor, but got {type(x)}.")
        _size = x.shape
        _dtype = x.dtype
        sign_matrix = self.sign(x)
        negative_filter = sign_matrix.clip(None, 0)
        positive_filter = sign_matrix.clip(0, None)
        mask = ops.uniform(_size, self.lower, self.upper).astype(_dtype)
        negative_mask = negative_filter * mask * -1
        total_mask = negative_mask + positive_filter
        out = total_mask * x
        return out


class SeLU(Cell):
    r"""
    Applies activation function SeLU (Scaled exponential Linear Unit) element-wise.

    SeLU Activation Function Graph:

    .. image:: ../images/SeLU.png
        :align: center

    Refer to :func:`mindspore.ops.selu` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> selu = nn.SeLU()
        >>> output = selu(input_x)
        >>> print(output)
        [[-1.1113307 4.202804 -1.7575096]
        [ 2.101402 -1.7462534 9.456309 ]]
    """

    def __init__(self):
        """Initialize SeLU"""
        super(SeLU, self).__init__()
        self.selu = ops.SeLU()

    def construct(self, input_x):
        return self.selu(input_x)


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
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> silu = nn.SiLU()
        >>> output = silu(input)
        >>> print(output)
        [-0.269  1.762  -0.1423  1.762  -0.269]
    """

    def __init__(self):
        """Initialize SiLU."""
        super(SiLU, self).__init__()

    def construct(self, x):
        return ops.function.silu(x)


class Tanh(Cell):
    r"""
    Applies the Tanh function element-wise, returns a new tensor with the hyperbolic tangent of the elements of input,
    The input is a Tensor with any valid shape.

    Tanh function is defined as:

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Tanh Activation Function Graph:

    .. image:: ../images/Tanh.png
        :align: center

    Inputs:
        - **x** (Tensor) - Tensor of any dimension, input with data type of float16 or float32.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([1, 2, 3, 2, 1]), mindspore.float16)
        >>> tanh = nn.Tanh()
        >>> output = tanh(x)
        >>> print(output)
        [0.7617 0.964  0.995  0.964  0.7617]
    """

    def __init__(self):
        """Initialize Tanh."""
        super(Tanh, self).__init__()
        self.tanh = ops.Tanh()

    def construct(self, x):
        return self.tanh(x)


class Tanhshrink(Cell):
    r"""
    Applies Tanhshrink activation function element-wise and returns a new tensor.

    Tanh function is defined as:

    .. math::
        tanhshrink(x_i) =x_i- \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)}
        = x_i-\frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **x** (Tensor) - Tensor of any dimension.

    Outputs:
        Tensor, with the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([1, 2, 3, 2, 1]), ms.float16)
        >>> tanhshrink = nn.Tanhshrink()
        >>> output = tanhshrink(x)
        >>> print(output)
        [0.2383 1.036  2.004  1.036  0.2383]
    """

    def __init__(self):
        """Initialize Tanhshrink."""
        super(Tanhshrink, self).__init__()

    def construct(self, x):
        return ops.tanhshrink(x)


class Hardtanh(Cell):
    r"""
    Applies the Hardtanh function element-wise. The activation function is defined as:

    .. math::
        \text{Hardtanh}(x) = \begin{cases}
            1, & \text{ if } x > 1; \\
            -1, & \text{ if } x < -1; \\
            x, & \text{ otherwise. }
        \end{cases}

    Linear region range :math:`[-1, 1]` can be adjusted using `min_val` and `max_val`.

    Hardtanh Activation Function Graph:

    .. image:: ../images/Hardtanh.png
        :align: center

    Note:
        On Ascend, data type of float16 might lead to accidental accuracy problem.

    Args:
        min_val (Union[int, float]): Minimum value of the linear region range. Default: ``-1.0`` .
        max_val (Union[int, float]): Maximum value of the linear region range. Default: ``1.0`` .

    Inputs:
        - **x** (Tensor) - Input Tensor with data type of float16 or float32.
          On CPU and Ascend support dimension 0-7D. On GPU support dimension 0-4D.

    Outputs:
        Tensor, with the same dtype and shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If dtype of `min_val` is neither float nor int.
        TypeError: If dtype of `max_val` is neither float nor int.
        ValueError: If `min_val` is not less than `max_val`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        >>> output = hardtanh(x)
        >>> print(output)
        [-1. -1.  0.  1.  1.]
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        """Initialize Hardtanh."""
        super(Hardtanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        if self.min_val >= self.max_val:
            raise ValueError(f"For Hardtanh, min_val should be less than max_val,"
                             f"but got {self.min_val} and {self.max_val}")

    def construct(self, x):
        return ops.hardtanh(x, self.min_val, self.max_val)


class GELU(Cell):
    r"""
    Applies GELU function to each element of the input. The input is a Tensor with any valid shape.

    GELU is defined as:

    .. math::

        GELU(x_i) = x_i*P(X < x_i),

    where :math:`P` is the cumulative distribution function
    of standard Gaussian distribution and :math:`x_i` is the element of the input.

    GELU Activation Function Graph:

    .. image:: ../images/GELU.png
        :align: center

    Args:
        approximate (bool, optional): Whether to enable approximation. Default: ``True`` .

            If `approximate` is ``True``, The gaussian error linear activation is:

            :math:`0.5 * x * (1 + tanh(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))`

            else, it is:

            :math:`x * P(X <= x) = 0.5 * x * (1 + erf(x / \sqrt(2)))`, where :math:`P(X) ~ N(0, 1)`.

    Note:
        - when calculating the input gradient of GELU with an input value of infinity, there are differences
          in the output of the backward between ``Ascend`` and ``GPU``.
        - when x is -inf, the computation result of ``Ascend`` is 0, and the computation result of ``GPU`` is Nan.
        - when x is inf, the computation result of ``Ascend`` is dy, and the computation result of ``GPU`` is Nan.
        - In mathematical terms, the result of Ascend has higher precision.

    Inputs:
        - **x** (Tensor) - The input of GELU with data type of float16, float32, or float64.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is not one of float16, float32, or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> gelu = nn.GELU()
        >>> output = gelu(x)
        >>> print(output)
        [[-1.5880802e-01  3.9999299e+00 -3.1077917e-21]
         [ 1.9545976e+00 -2.2918017e-07  9.0000000e+00]]
        >>> gelu = nn.GELU(approximate=False)
        >>> # CPU not support "approximate=False", using "approximate=True" instead
        >>> output = gelu(x)
        >>> print(output)
        [[-1.5865526e-01  3.9998732e+00 -0.0000000e+00]
         [ 1.9544997e+00 -1.4901161e-06  9.0000000e+00]]
    """

    def __init__(self, approximate=True):
        """Initialize GELU."""
        super(GELU, self).__init__()
        validator.check_bool(approximate, 'approximate', self.cls_name)
        self.approximate = 'tanh'
        if not approximate:
            self.approximate = 'none'

    def construct(self, x):
        return ops.gelu(x, approximate=self.approximate)


class FastGelu(Cell):
    r"""
    Applies FastGelu function to each element of the input. The input is a Tensor with any valid shape.

    FastGelu is defined as:

    .. math::
        FastGelu(x_i) = \frac {x_i} {1 + \exp(-1.702 * \left| x_i \right|)} *
                           \exp(0.851 * (x_i - \left| x_i \right|))

    where :math:`x_i` is the element of the input.

    FastGelu Activation Function Graph:

    .. image:: ../images/FastGelu.png
        :align: center

    Inputs:
        - **x** (Tensor) - The input of FastGelu with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> fast_gelu = nn.FastGelu()
        >>> output = fast_gelu(x)
        >>> print(output)
        [[-1.5418735e-01  3.9921875e+00 -9.7473649e-06]
         [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]]
    """

    def __init__(self):
        """Initialize FastGelu."""
        super(FastGelu, self).__init__()
        self.fast_gelu = ops.FastGeLU()

    def construct(self, x):
        return self.fast_gelu(x)


class Sigmoid(Cell):
    r"""
    Applies sigmoid activation function element-wise.

    Sigmoid function is defined as:

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    where :math:`x_i` is the element of `x`.

    Sigmoid Activation Function Graph:

    .. image:: ../images/Sigmoid.png
        :align: center

    Inputs:
        - **input** (Tensor) - `input` is :math:`x` in the preceding formula. Tensor of any dimension,
          the data type is float16, float32, float64, complex64 or complex128.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If dtype of `input` is not float16, float32, float64, complex64 or complex128.
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> sigmoid = nn.Sigmoid()
        >>> output = sigmoid(input)
        >>> print(output)
        [0.2688  0.11914 0.5     0.881   0.7305 ]
    """

    def __init__(self):
        """Initialize Sigmoid."""
        super(Sigmoid, self).__init__()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        return self.sigmoid(x)


class Softsign(Cell):
    r"""
    Applies softsign activation function element-wise.

    Softsign Activation Function Graph:

    .. image:: ../images/Softsign.png
        :align: center

    Refer to :func:`mindspore.ops.softsign` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([0, -1, 2, 30, -30]), mindspore.float32)
        >>> softsign = nn.Softsign()
        >>> output = softsign(x)
        >>> print(output)
        [ 0.        -0.5         0.6666667  0.9677419 -0.9677419]
    """

    def __init__(self):
        """Initialize Softsign."""
        super(Softsign, self).__init__()
        self.softsign = ops.Softsign()

    def construct(self, x):
        return self.softsign(x)


class PReLU(Cell):
    r"""
    Applies PReLU activation function element-wise.

    PReLU is defined as:

    .. math::

        PReLU(x_i)= \max(0, x_i) + w * \min(0, x_i),

    where :math:`x_i` is an element of an channel of the input.

    Here :math:`w` is a learnable parameter with a default initial value ``0.25``.
    Parameter :math:`w` has dimensionality of the argument channel. If called without argument
    channel, a single parameter :math:`w` will be shared across all channels.

    PReLU Activation Function Graph:

    .. image:: ../images/PReLU.png
        :align: center

    Args:
        channel (int, optional): The elements number of parameter :math:`w`.
          It could be an int, and the value is ``1`` or the channels number of input tensor `x`. Default: ``1`` .
        w (Union[float, list, Tensor], optional): The initial value of parameter. It could be a float, a float list or
          a tensor has the same dtype as the input tensor `x`. Default: ``0.25`` .

    Inputs:
        - **x** (Tensor) - The input of PReLU with data type of float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same dtype and shape as the `x`.

    Raises:
        TypeError: If `channel` is not an int.
        TypeError: If `w` is not one of a float, a list[float], a Tensor[float].
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If the `x` is a 0-D or 1-D Tensor on Ascend.
        ValueError: If `channel` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[[[0.1, 0.6], [0.9, 0.9]]]]), mindspore.float32)
        >>> prelu = nn.PReLU()
        >>> output = prelu(x)
        >>> print(output)
        [[[[0.1 0.6]
           [0.9 0.9]]]]

    """

    @cell_attr_register(attrs="")
    def __init__(self, channel=1, w=0.25):
        """Initialize PReLU."""
        super(PReLU, self).__init__()
        validator.check_positive_int(channel, 'channel', self.cls_name)
        if isinstance(w, (float, np.float32)):
            tmp = np.empty((channel,), dtype=np.float32)
            tmp.fill(w)
            w = Tensor(tmp, dtype=mstype.float32)
        elif isinstance(w, list):
            if len(w) != channel:
                raise ValueError(f"For '{self.cls_name}', the length of 'w' must be equal to the 'channel' when "
                                 f"the 'w' is a list, but got the length of 'w': {len(w)}, the 'channel': {channel}.")

            for i in w:
                if not isinstance(i, (float, np.float32)):
                    raise ValueError(f"For '{self.cls_name}', all elements in 'w' must be "
                                     f"float when the 'w' is a list, but got {i}.")
            w = Tensor(w, dtype=mstype.float32)
        elif isinstance(w, Tensor):
            if w.dtype not in (mstype.float16, mstype.float32):
                raise ValueError(f"For '{self.cls_name}', the dtype of 'w' must be float16 or "
                                 f"float32 when the 'w' is a tensor, but got {w.dtype}.")
            if len(w.shape) != 1 or w.shape[0] != channel:
                raise ValueError(f"For '{self.cls_name}', the dimension of 'w' must be 1, and the elements number "
                                 f"should be equal to the 'channel' when the 'w' is a tensor, "
                                 f"but got 'w' shape {w.shape}, the 'channel' {channel}.")
        else:
            raise TypeError(f"For '{self.cls_name}', the 'w' only supported float, list and tensor, "
                            f"but got {type(w).__name__}.")
        self.w = Parameter(w, name='a')
        self.prelu = ops.PReLU()

    def construct(self, x):
        return self.prelu(x, ops.cast(self.w, x.dtype))


class PReLUExt(Cell):
    r"""
    Applies PReLU activation function element-wise.

    PReLU is defined as:

    .. math::

        PReLU(x_i)= \max(0, x_i) + w * \min(0, x_i),

    where :math:`x_i` is an element of an channel of the input.

    Here :math:`w` is a learnable parameter with a default initial value 0.25.
    Parameter :math:`w` has dimensionality of the argument channel. If called without argument
    channel, a single parameter :math:`w` will be shared across all channels.

    PReLU Activation Function Graph:

    .. image:: ../images/PReLU2.png
        :align: center

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int, optional): number of `w` to learn. Although it takes an int as input,
            there is only two legitimate values: 1, or the number of channels at Tensor `input`. Default: ``1`` .
        init (float, optional): the initial value of `w`. Default: ``0.25`` .
        dtype (mindspore.dtype, optional): the type of `w`. Default: ``None`` . Supported data type
            is {float16, float32, bfloat16}.

    Inputs:
        - **input** (Tensor) - The input of PReLU.

    Outputs:
        Tensor, with the same dtype and shape as the `input`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[[[0.1, 0.6], [0.9, 0.9]]]]), mindspore.float32)
        >>> prelu = nn.PReLUExt()
        >>> output = prelu(x)
        >>> print(output)
        [[[[0.1 0.6]
           [0.9 0.9]]]]

    """

    def __init__(self, num_parameters=1, init=0.25, dtype=None):
        """Initialize PReLUExt."""
        super(PReLUExt, self).__init__()
        tmp = np.empty((num_parameters,), dtype=np.float32)
        tmp.fill(init)
        w = Tensor(tmp, dtype=dtype)
        self.weight = Parameter(w, name='weight')

    def construct(self, input):
        return ops.prelu(input, self.weight)


class HSwish(Cell):
    r"""
    Applies Hard Swish activation function element-wise.

    Hard swish is defined as:

    .. math::
        \text{HSwish}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        input, & \text{ if } input \geq +3, \\
        input*(input + 3)/6, & \text{ otherwise }
        \end{cases}

    HSwish Activation Function Graph:

    .. image:: ../images/HSwish.png
        :align: center

    Inputs:
        - **input** (Tensor) - The input of HSwish.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If `input` is neither int nor float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> hswish = nn.HSwish()
        >>> result = hswish(input)
        >>> print(result)
        [-0.3333 -0.3333  0.      1.667   0.6665]
    """

    def __init__(self):
        """Initialize HSwish."""
        super(HSwish, self).__init__()
        self.hswish = ops.HSwish()

    def construct(self, input):
        return self.hswish(input)


class HSigmoid(Cell):
    r"""
    Applies Hard Sigmoid activation function element-wise.

    Hard Sigmoid is defined as:

    .. math::
        \text{HSigmoid}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        1, & \text{ if } input \geq +3, \\
        input/6 + 1/2, & \text{ otherwise }
        \end{cases}

    HSigmoid Activation Function Graph:

    .. image:: ../images/HSigmoid.png
        :align: center

    Inputs:
        - **input** (Tensor) - The input of HSigmoid.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `input` is neither int nor float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> hsigmoid = nn.HSigmoid()
        >>> result = hsigmoid(input)
        >>> print(result)
        [0.3333 0.1666 0.5    0.8335 0.6665]
    """

    def __init__(self):
        """Initialize HSigmoid."""
        super(HSigmoid, self).__init__()
        self.hsigmoid = ops.HSigmoid()

    def construct(self, input):
        return self.hsigmoid(input)


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

    Inputs:
        - **x** (Tensor) - The input of LogSigmoid with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.LogSigmoid()
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> output = net(x)
        >>> print(output)
        [-0.31326166 -0.12692806 -0.04858734]
    """

    def __init__(self):
        """Initialize LogSigmoid."""
        super(LogSigmoid, self).__init__()
        self.mul = ops.Mul()
        self.exp = ops.Exp()
        self.add = ops.Add()
        self.rec = ops.Reciprocal()
        self.log = ops.Log()

    def construct(self, input_x):
        neg_input = self.mul(input_x, -1)
        exp_neg_input = self.exp(neg_input)
        exp_neg_input_1 = self.add(exp_neg_input, 1)
        rec_exp_neg_input_1 = self.rec(exp_neg_input_1)
        ret = self.log(rec_exp_neg_input_1)
        return ret


class LRN(Cell):
    r"""
    Local Response Normalization.

    .. warning::
        LRN is deprecated on Ascend due to potential accuracy problem. It's recommended to use other
        normalization methods, e.g. :class:`mindspore.nn.BatchNorm1d` ,
        :class:`mindspore.nn.BatchNorm2d` , :class:`mindspore.nn.BatchNorm3d`.

    Refer to :func:`mindspore.ops.lrn` for more details.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input_x = Tensor(np.array([[[[0.1], [0.2]],
        ...                       [[0.3], [0.4]]]]), mindspore.float32)
        >>> output = nn.LRN()(input_x)
        >>> print(output)
        [[[[0.09534626]
           [0.1825742 ]]
          [[0.2860388 ]
           [0.3651484 ]]]]
    """

    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS"):
        """Initialize LRN."""
        super(LRN, self).__init__()
        self.lrn_op = NN_OPS.LRN(depth_radius, bias, alpha, beta, norm_region)

    def construct(self, input_x):
        return self.lrn_op(input_x)


class SoftShrink(Cell):
    r"""
    Applies the SoftShrink function element-wise.

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    SoftShrink Activation Function Graph:

    .. image:: ../images/Softshrink.png
        :align: center

    Args:
        lambd (number, optional): The threshold :math:`\lambda` defined by the Soft Shrink formula.
            It should be greater than or equal to 0, default: ``0.5`` .

    Inputs:
        - **input** (Tensor) - The input of Soft Shrink. Supported dtypes:

          - Ascend: float16, float32, bfloat16.
          - CPU/GPU: float16, float32.

    Outputs:
        Tensor, the same shape and data type as the input.

    Raises:
        TypeError: If `lambd` is not a float, int or bool.
        TypeError: If `input` is not a tensor.
        TypeError: If dtype of `input` is not float16, float32 or bfloat16.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([[ 0.5297,  0.7871,  1.1754], [ 0.7836,  0.6218, -1.1542]]), mindspore.float16)
        >>> softshrink = nn.SoftShrink()
        >>> output = softshrink(input)
        >>> print(output)
        [[ 0.02979  0.287    0.676  ]
         [ 0.2837   0.1216  -0.6543 ]]
    """

    def __init__(self, lambd=0.5):
        super(SoftShrink, self).__init__()
        self.softshrink = ops.SoftShrink(lambd)

    def construct(self, input):
        output = self.softshrink(input)
        return output


class HShrink(Cell):
    r"""
    Applies Hard Shrink activation function element-wise.

    The formula is defined as follows:

    .. math::
        \text{HShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    HShrink Activation Function Graph:

    .. image:: ../images/HShrink.png
        :align: center

    Args:
        lambd (number, optional): The threshold :math:`\lambda` defined by the Hard Shrink formula. Default: ``0.5`` .

    Inputs:
        - **input** (Tensor) - The input of Hard Shrink. Supported dtypes:

          - Ascend: float16, float32, bfloat16.
          - CPU/GPU: float16, float32.

    Outputs:
        Tensor, the same shape and data type as the input.

    Raises:
        TypeError: If `lambd` is not a float, int or bool.
        TypeError: If `input` is not a tensor.
        TypeError: If dtype of `input` is not float16, float32 or bfloat16.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]), mindspore.float32)
        >>> hshrink = nn.HShrink()
        >>> output = hshrink(input)
        >>> print(output)
        [[ 0.      1.      2.    ]
         [ 0.      0.     -2.1233]]
    """

    def __init__(self, lambd=0.5):
        super(HShrink, self).__init__()
        self.hshrink = ops.HShrink(lambd)

    def construct(self, input):
        return self.hshrink(input)


class Threshold(Cell):
    r"""
    Thresholds each element of the input Tensor.

    The formula is defined as follows:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold (Union[int, float]): The value to threshold at.
        value (Union[int, float]): The value to replace with when element is less than threshold.

    Inputs:
        - **input_x** (Tensor) - The input of Threshold with data type of float16 or float32.

    Outputs:
        Tensor, the same shape and data type as the `input_x`.

    Raises:
        TypeError: If `threshold` is not a float or an int.
        TypeError: If `value` is not a float or an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> m = nn.Threshold(0.1, 20)
        >>> inputs = Tensor([0.1, 0.2, 0.3], mindspore.float32)
        >>> outputs = m(inputs)
        >>> print(outputs)
        [ 20.0     0.2      0.3]
    """

    def __init__(self, threshold, value):
        """Initialize Threshold."""
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value

    def construct(self, input_x):
        return ops.threshold(input_x, self.threshold, self.value)


class Mish(Cell):
    r"""
    Computes MISH (A Self Regularized Non-Monotonic Neural Activation Function)
    of input tensors element-wise.

    Refer to :func:`mindspore.ops.mish` for more details.

    Mish Activation Function Graph:

    .. image:: ../images/Mish.png
        :align: center

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> mish = nn.Mish()
        >>> output = mish(x)
        >>> print(output)
        [[-3.03401530e-01 3.99741292e+00 -2.68321624e-03]
         [ 1.94395900e+00 -3.35762873e-02 9.00000000e+00]]
    """

    def __init__(self):
        """Initialize Mish."""
        super().__init__("Mish")
        self.mish = NN_OPS.Mish()

    def construct(self, input_x):
        return self.mish(input_x)


class GLU(Cell):
    r"""
    The gated linear unit function.

    .. math::
        {GLU}(a, b)= a \otimes \sigma(b)

    where :math:`a` is the first half of the input matrices and :math:`b` is the second half.

    Here :math:`\sigma` is the sigmoid function, and :math:`\otimes` is the Hadamard product.

    Args:
        axis (int, optional): the axis to split the input. Default: ``-1`` , the last axis in `x`.

    Inputs:
        - **x** (Tensor) - :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional dimensions.

    Outputs:
        Tensor, the same dtype as the `x`, with the shape :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> m = ms.nn.GLU()
        >>> input = ms.Tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]])
        >>> output = m(input)
        >>> print(output)
        [[0.05744425 0.11973753]
         [0.33409387 0.41398472]]
    """

    def __init__(self, axis=-1):
        """Initialize GLU."""
        super().__init__("GLU")
        self.dim = axis
        self.spilt = ops.Split(axis=axis, output_num=2)
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        x1, x2 = self.spilt(x)
        x2 = self.sigmoid(x2)
        return x1 * x2


_activation = {
    'softmin': Softmin,
    'softmax': Softmax,
    'softmax2d': Softmax2d,
    'logsoftmax': LogSoftmax,
    'logsoftmaxExt': LogSoftmaxExt,
    'relu': ReLU,
    'relu6': ReLU6,
    'rrelu': RReLU,
    'silu': SiLU,
    'tanh': Tanh,
    'tanhshrink': Tanhshrink,
    'hardtanh': Hardtanh,
    'gelu': GELU,
    'fast_gelu': FastGelu,
    'elu': ELU,
    'sigmoid': Sigmoid,
    'softsign': Softsign,
    'prelu': PReLU,
    'preluExt': PReLUExt,
    'leakyrelu': LeakyReLU,
    'hswish': HSwish,
    'hsigmoid': HSigmoid,
    'logsigmoid': LogSigmoid,
    'softshrink': SoftShrink,
    'hshrink': HShrink,
    'threshold': Threshold,
    'mish': Mish,
}


def get_activation(name, prim_name=None):
    """
    Gets the activation function.

    Args:
        name (str): The name of the activation function.
        prim_name (Union[str, None]): The name of primitive. Default: ``None`` .

    Returns:
        Function, the activation function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> sigmoid = nn.get_activation('sigmoid')
        >>> print(sigmoid)
        Sigmoid()
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if name is None:
        return None

    if name not in _activation:
        raise KeyError(f"{msg_prefix} 'name' must be in {list(_activation.keys())}, but got {name}.")
    return _activation[name]()
