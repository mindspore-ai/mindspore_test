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
"""
Neural Networks Cells.

Predefined building blocks or computing units to construct neural networks.
"""
from __future__ import absolute_import
import mindspore.ops as ops
from mindspore.mint.nn import functional as F
from mindspore.nn.cell import Cell
from mindspore.nn import EmbeddingExt as Embedding, MaxPool2dExt as MaxPool2d, LayerNormExt as LayerNorm, Linear

# 1

# 2

# 3
from mindspore.nn.layer.basic import Identity
# 4

# 5

# 6
from mindspore.nn.layer.basic import UnfoldExt as Unfold
# 7
from mindspore.nn.layer.basic import Fold
# 8
from mindspore.nn.layer.activation import SoftmaxExt as Softmax
# 9
from mindspore.nn.layer.basic import UpsampleExt as Upsample
# 10

# 11
from mindspore.nn.layer import ReLU

# 12

# 13

# 14
from mindspore.nn.layer.basic import DropoutExt as Dropout
# 15

# 16
from mindspore.nn.layer import LogSoftmaxExt as LogSoftmax
# 17

# 18
from mindspore.nn.layer import PReLUExt as PReLU
# 19

# 20

# 21

# 22

# 23

# 24

# 25

# 26

# 27

# 28

# 29

# 30

# 31

# 32

# 33

# 34

# 35

# 36

# 37

# 38

# 39

# 40
from mindspore.mint.nn.layer.normalization import GroupNorm
from mindspore.mint.nn.layer.normalization import LayerNorm
# 41

# 42

# 43

# 44

# 45

# 46
from mindspore.mint.nn.layer.activation import SiLU, LogSigmoid

# 47

# 48

# 49

# 50

# 51

# 52

# 53

# 54

# 55

# 56

# 57

# 58

# 59

# 60

# 61

# 62

# 63

# 64

# 65

# 66

# 67

# 68

# 69

# 70

# 71

# 72

# 73

# 74

# 75

# 76

# 77

# 78

# 79

# 80

# 81

# 82

# 83

# 84

# 85

# 86

# 87

# 88

# 89

# 90

# 91

# 92

# 93

# 94

# 95

# 96

# 97

# 98

# 99
from mindspore.nn.layer import AvgPool2dExt as AvgPool2d
# 100
from mindspore.nn.layer import SoftShrink as Softshrink
# 159

# 220
from mindspore.nn.layer import HShrink as Hardshrink
# 221
from mindspore.nn.layer import HSigmoid as Hardsigmoid
# 222
from mindspore.nn.layer import HSwish as Hardswish
# 238
from mindspore.nn.loss import L1LossExt as L1Loss

# 257

# 258
from mindspore.ops.function.nn_func import mse_loss_ext


# 674
from mindspore.mint.nn.layer.normalization import BatchNorm1d

# 675
from mindspore.mint.nn.layer.normalization import BatchNorm2d

# 676
from mindspore.mint.nn.layer.normalization import BatchNorm3d

from mindspore.mint.nn.layer.pooling import AdaptiveAvgPool1d

from mindspore.mint.nn.layer.pooling import AdaptiveAvgPool2d


class BCEWithLogitsLoss(Cell):
    r"""
    Adds sigmoid activation function to `input` as logits, and uses this logits to compute binary cross entropy
    between the logits and the target.

    Sets input `input` as :math:`X`, input `target` as :math:`Y`, output as :math:`L`. Then,

    .. math::
        p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}}

    .. math::
        L_{ij} = -[Y_{ij} \cdot \log(p_{ij}) + (1 - Y_{ij}) \cdot \log(1 - p_{ij})]

    Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            If not None, it can be broadcast to a tensor with shape of `target`, data type must be float16, float32 or
            bfloat16(only Atlas A2 series products are supported). Default: ``None`` .
        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the weighted mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector with length equal to the
            number of classes. If not None, it must be broadcast to a tensor with shape of `input`, data type
            must be float16, float32 or bfloat16(only Atlas A2 series products are supported). Default: ``None`` .

    Inputs:
        - **input** (Tensor) - Input `input` with shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions. The data type must be float16, float32 or bfloat16(only Atlas A2 series products
          are supported).
        - **target** (Tensor) - Ground truth label with shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions. The same shape and data type as `input`.

    Outputs:
        Tensor or Scalar, if `reduction` is ``'none'``, its shape is the same as `input`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If input `input` or `target` is not Tensor.
        TypeError: If `weight` or `pos_weight` is a parameter.
        TypeError: If data type of `reduction` is not string.
        ValueError: If `weight` or `pos_weight` can not be broadcast to a tensor with shape of `input`.
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'``, ``'sum'``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import mint
        >>> import numpy as np
        >>> input = ms.Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
        >>> target = ms.Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
        >>> loss = mint.nn.BCEWithLogitsLoss()
        >>> output = loss(input, target)
        >>> print(output)
        0.3463612
    """

    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce_with_logits = ops.auto_generate.BCEWithLogitsLoss(reduction)
        self.weight = weight
        self.pos_weight = pos_weight

    def construct(self, input, target):
        out = self.bce_with_logits(input, target, self.weight, self.pos_weight)
        return out


class SELU(Cell):
    r"""
    Activation function SELU (Scaled exponential Linear Unit).

    Refer to :func:`mindspore.mint.nn.functional.selu` for more details.

    SELU Activation Function Graph:

    .. image:: ../images/SeLU.png
        :align: center

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> selu = mint.nn.SELU()
        >>> output = selu(input)
        >>> print(output)
        [[-1.1113307 4.202804 -1.7575096]
        [ 2.101402 -1.7462534 9.456309 ]]
    """

    def __init__(self):
        """Initialize SELU"""
        super(SELU, self).__init__()

    def construct(self, input):
        return F.selu(input)


class GELU(Cell):
    r"""
    Activation function GELU (Gaussian Error Linear Unit).

    Refer to :func:`mindspore.mint.nn.functional.gelu` for more details.

    GELU Activation Function Graph:

    .. image:: ../images/GELU.png
        :align: center

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> gelu = mint.nn.GELU()
        >>> output = gelu(input)
        >>> print(output)
        [[-1.5880802e-01  3.9999299e+00 -3.1077917e-21]
         [ 1.9545976e+00 -2.2918017e-07  9.0000000e+00]]
        >>> gelu = mint.nn.GELU(approximate=False)
        >>> # CPU not support "approximate=False", using "approximate=True" instead
        >>> output = gelu(input)
        >>> print(output)
        [[-1.5865526e-01  3.9998732e+00 -0.0000000e+00]
         [ 1.9544997e+00 -1.4901161e-06  9.0000000e+00]]
    """

    def __init__(self):
        """Initialize GELU"""
        super(GELU, self).__init__()

    def construct(self, input):
        return F.gelu(input)


class Mish(Cell):
    r"""
    Computes MISH (A Self Regularized Non-Monotonic Neural Activation Function)
    of input tensors element-wise.

    Refer to :func:`mindspore.mint.nn.functional.mish` for more details.

    Mish Activation Function Graph:

    .. image:: ../images/Mish.png
        :align: center

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.1, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> mish = mint.nn.Mish()
        >>> output = mish(x)
        >>> print(output)
        [[-3.0764845e-01 3.9974124e+00 -2.6832507e-03]
         [ 1.9439589e+00 -3.3576239e-02 8.9999990e+00]]
    """

    def __init__(self):
        """Initialize Mish."""
        super(Mish, self).__init__()

    def construct(self, input):
        return F.mish(input)


class MSELoss(Cell):
    r"""
    Calculates the mean squared error between the predicted value and the label value.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the unreduced loss (i.e. with argument reduction set to 'none') of :math:`x` and :math:`y` is given as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with} \quad l_n = (x_n - y_n)^2.

    where :math:`N` is the batch size. If `reduction` is not ``'none'``, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

    Inputs:
        - **logits** (Tensor) - The predicted value of the input. Tensor of any dimension.
          The data type needs to be consistent with the `labels`. It should also be broadcastable with the `labels`.
        - **labels** (Tensor) - The input label. Tensor of any dimension.
          The data type needs to be consistent with the `logits`. It should also be broadcastable with the `logits`.

    Outputs:
        - Tensor. If `reduction` is ``'mean'`` or ``'sum'``, the shape of output is `Tensor Scalar`.
        - If reduction is ``'none'``, the shape of output is the broadcasted shape of `logits` and `labels` .

    Raises:
        ValueError: If `reduction` is not one of ``'mean'``, ``'sum'`` or ``'none'``.
        ValueError: If `logits` and `labels` are not broadcastable.
        TypeError: If `logits` and `labels` are in different data type.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> # Case 1: logits.shape = labels.shape = (3,)
        >>> loss = nn.MSELoss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 1, 1]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        1.6666667
        >>> # Case 2: logits.shape = (3,), labels.shape = (2, 3)
        >>> loss = nn.MSELoss(reduction='none')
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [[0. 1. 4.]
         [0. 0. 1.]]
    """

    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.mse_loss = mse_loss_ext
        self.reduction = reduction

    def construct(self, input, target):
        out = self.mse_loss(input, target, self.reduction)
        return out


__all__ = [
    # 1
    'BCEWithLogitsLoss',
    # 2

    # 3
    'Identity',
    # 4

    # 5

    # 6
    'Fold',
    # 7
    'Unfold',
    # 8
    'Softmax',
    # 9
    'Upsample',
    # 10

    # 11
    'ReLU',

    # 12

    # 13

    # 14

    # 15

    # 16
    'LogSoftmax',
    # 17

    # 18
    'PReLU',
    # 19

    # 20

    # 21

    # 22

    # 23

    # 24

    # 25

    # 26

    # 27

    # 28

    # 29

    # 30

    # 31

    # 32

    # 33

    # 34

    # 35

    # 36

    # 37

    # 38
    'Linear',
    # 39

    # 40
    'GroupNorm',

    # 41

    # 42

    # 43

    # 44

    # 45

    # 46
    'SiLU',

    # 47

    # 48

    # 49

    # 50

    # 51

    # 52

    # 53

    # 54

    # 55

    # 56

    # 57

    # 58

    # 59

    # 60

    # 61

    # 62

    # 63

    # 64

    # 65

    # 66

    # 67

    # 68

    # 69

    # 70

    # 71

    # 72

    # 73

    # 74

    # 75

    # 76

    # 77

    # 78

    # 79

    # 80

    # 81

    # 82

    # 83

    # 84

    # 85

    # 86

    # 87

    # 88

    # 89

    # 90

    # 91

    # 92

    # 93

    # 94

    # 95

    # 96
    'AdaptiveAvgPool1d',

    # 97
    'AdaptiveAvgPool2d',

    # 98

    # 99
    'AvgPool2d',
    # 100
    'SELU',
    # 159
    'GELU',
    # 220
    'Hardshrink',
    # 221
    'Hardsigmoid',
    # 222
    'Hardswish',
    # 238
    'L1Loss',
    # 267
    'Mish',
    # 258
    'MSELoss',
    # 259

    # 556
    'LogSigmoid',

    # 674
    'BatchNorm1d',
    # 675
    'BatchNorm2d',
    # 676
    'BatchNorm3d',
]
