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
"""mint nn functional."""
from __future__ import absolute_import
import mindspore.ops as ops
import mindspore.mint as mint
from mindspore import log as logger
from mindspore import _checkparam as validator
from mindspore.ops.primitive import constexpr
from mindspore.ops.function.nn_func import max_pool2d_ext as max_pool2d
from mindspore.ops.functional import (
    conv_transpose2d,
    grid_sample
)
# 1

# 2

# 3

# 4
from mindspore.ops.function.nn_func import interpolate_ext as interpolate
# 5
from mindspore.ops.function.nn_func import pad_ext as pad
# 6
from mindspore.ops.auto_generate import unfold_ext as unfold
# 7
from mindspore.ops.auto_generate import fold_ext as fold
# 8
from mindspore.ops.functional import layer_norm
# 9

# 10

# 11
from mindspore.ops.functional import relu

from mindspore.ops.function.nn_func import relu_

# 12

# 13

# 14
from mindspore.ops.function.nn_func import dropout_ext as dropout
# 15
from mindspore.ops.function.nn_func import conv2d_ext as conv2d
# 16
from mindspore.ops.function.nn_func import log_softmax_ext as log_softmax
# 18
from mindspore.ops.auto_generate import prelu
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
from mindspore.ops.function.nn_func import softmax_ext as softmax

# 32

# 33

# 34
from mindspore.ops.function.nn_func import batch_norm_ext as batch_norm
# 35

# 36
from mindspore.ops.functional import gelu
# 37

# 38
from mindspore.ops.functional import dense as linear
# 39
from mindspore.ops.functional import group_norm
# 40

# 41

# 42

# 43

# 44

# 45

# 46
from mindspore.ops.functional import silu
# 47

# 48

# 49
from mindspore.ops.functional import sigmoid
# 50

# 51

# 52
from mindspore.ops.functional import embedding
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
from mindspore.ops.function.nn_func import avg_pool2d_ext as avg_pool2d
# 91

# 92
from mindspore.ops.auto_generate import leaky_relu_ext as leaky_relu
# 93
from mindspore.ops.auto_generate import softplus_ext as softplus  # pylint: disable=W0611
# 94
from mindspore.ops.function.math_func import tanh
# 95

# 96

# 97

# 98

# 99
from mindspore.ops.auto_generate import selu_ext as selu  # pylint: disable=W0611
# 100
from mindspore.ops.auto_generate import softshrink  # pylint: disable=W0611
# 220
from mindspore.ops.function.nn_func import hardshrink  # pylint: disable=W0611
# 221
from mindspore.ops.function.nn_func import hardsigmoid  # pylint: disable=W0611
# 222
from mindspore.ops.function.nn_func import hardswish  # pylint: disable=W0611
# 267
from mindspore.ops.auto_generate import mish_ext as mish  # pylint: disable=W0611
# 238
from mindspore.ops.auto_generate import l1_loss_ext as l1_loss  # pylint: disable=W0611

# 257

# 258
from mindspore.ops.function.nn_func import mse_loss_ext as mse_loss
# 259

# 323

# 324
from mindspore.ops.auto_generate import elu_ext as elu

# 426
from mindspore.ops.function.clip_func import clamp
# 427
from mindspore.ops.function.math_func import norm_ext
# 428
from mindspore.ops.functional import broadcast_to

# 556
from mindspore.ops.function.nn_func import logsigmoid_ext as logsigmoid

from mindspore.ops.auto_generate import adaptive_avg_pool1d

from mindspore.ops.functional import adaptive_avg_pool2d_ext as adaptive_avg_pool2d
from mindspore.ops.function.nn_func import cross_entropy_ext as cross_entropy


def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    r"""
    Computes the binary cross entropy(Measure the difference information between two probability distributions) between
    predictive value `input` and target value `target`.

    Set `input` as :math:`x`, `target` as :math:`y`, output as :math:`\ell(x, y)`, the
    weight of nth batch of binary cross entropy is :math:`w_n`.
    Let,

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    In which, :math:`L` indicates the loss of all `batch_size`, :math:`l` indicates the loss of one `batch_size`,
    and :math:`n` indicates one `batch_size` in the :math:`1-N` range. Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    .. warning::
        - The value of `input` must range from `0` to `l`.

    Args:
        input (Tensor): The predictive value whose data type must be float16 or float32.
        target (Tensor): The target value which has the same shape and data type as `input`.
            And the data type is float16 or float32.
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            Its shape must be able to broadcast to that of `input` and `target`.
            And it must have the same shape and data type as `input`. Default: ``None`` . If set to ``None`` ,
            the loss function
            will not consider any sample weights, and each sample will be treated as having equal importance
            when calculating the loss.
        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the weighted mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

    Returns:
        Tensor or Scalar. Returns Tensor that has the same dtype and shape as `input` if `reduction` is ``'none'``.
        Otherwise, returns a scalar Tensor.

    Raises:
        TypeError: If `input`, `target` or `weight` is not a Tensor.
        TypeError: If dtype of `input`, `target` or `weight` (if given) is neither float16 nor float32.
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.
        ValueError: If shape of `target` is not the same as `input` or `weight` (if given).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
        >>> target = Tensor(np.array([0., 1., 0.]), mindspore.float32)
        >>> weight = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = mint.nn.functional.binary_cross_entropy(input, target, weight)
        >>> print(output)
        0.38240486
    """
    return ops.function.binary_cross_entropy(input, target, weight, reduction)


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    r"""
    Adds sigmoid activation function to `input` as logits, and uses this logits to compute binary cross entropy
    between the logits and the target.
    Consistent with the function of `mindspore.ops.binary_cross_entropy_with_logits` .

    Sets input `input` as :math:`X`, input `target` as :math:`Y`, input `weight` as :math:`W`, output as :math:`L`.
    Then,

    .. math::

        \begin{array}{ll} \\
            p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}} \\
            L_{ij} = -[Y_{ij}log(p_{ij}) + (1 - Y_{ij})log(1 - p_{ij})]
        \end{array}

    :math:`i` indicates the :math:`i^{th}` sample, :math:`j` indicates the category. Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`\ell` indicates the method of calculating the loss. There are three methods:
    the first method is to provide the loss value directly,
    the second method is to calculate the average value of all losses,
    and the third method is to calculate the sum of all losses.

    This operator will multiply the output by the corresponding weight.
    The tensor :math:`weight` assigns different weights to each piece of data in the batch,
    and the tensor :math:`pos\_weight` adds corresponding weights to the positive examples of each category.

    In addition, it can trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:

    .. math::
        \begin{array}{ll} \\
            p_{ij,c} = sigmoid(X_{ij,c}) = \frac{1}{1 + e^{-X_{ij,c}}} \\
            L_{ij,c} = -[P_{c}Y_{ij,c} * log(p_{ij,c}) + (1 - Y_{ij,c})log(1 - p_{ij,c})]
        \end{array}

    where c is the class number (c>1 for multi-label binary classification, c=1 for single-label binary classification),
    n is the number of the sample in the batch and :math:`P_c` is the weight of the positive answer for the class c.
    :math:`P_c>1` increases the recall, :math:`P_c<1` increases the precision.

    Args:
        input (Tensor): Input `input` with shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions. The data type must be float16, float32 or bfloat16(only Atlas A2 series products
          are supported).
        target (Tensor): Ground truth label, has the same shape as `input`.
          The data type must be float16, float32 or bfloat16(only Atlas A2 series products are supported).
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element. It can be
          broadcast to a tensor with shape of `input`. Data type must be float16, float32 or bfloat16(only
          Atlas A2 series products are supported).
          Default: ``None``, `weight` is a Tensor whose value is ``1``.
        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the weighted mean of elements in the output.
            - ``'sum'``: the output elements will be summed.
        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector with length equal to the
          number of classes. It can be broadcast to a tensor with shape of `input`.
          Data type must be float16, float32 or bfloat16(only Atlas A2 series products are supported).
          Default: ``None``, it equals to `pos_weight` is a Tensor whose value is ``1``.

    Returns:
        Tensor or Scalar, if `reduction` is ``'none'``, it's a tensor with the same shape and type as input `input`.
        Otherwise, the output is a Scalar.

    Raises:
        TypeError: If input `input`, `target`, `weight`, `pos_weight` is not Tensor.
        TypeError: If data type of input `reduction` is not string.
        ValueError: If `weight` or `pos_weight` can not be broadcast to a tensor with shape of `input`.
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float32)
        >>> target = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float32)
        >>> weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> output = mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, 'mean', pos_weight)
        >>> print(output)
        0.3463612
    """
    return ops.function.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction)


def one_hot(tensor, num_classes=-1):
    r"""
    Computes a one-hot tensor.

    The locations represented by tensor in `tensor` take value `1`, while all
    other locations take value `0`.

    Args:
        tensor (Tensor): A tensor of indices. Tensor of shape :math:`(X_0, \ldots, X_n)`.
            Data type must be int32 or int64.
        num_classes (int): A scalar defining the depth of the one-hot dimension, default: ``-1``.

    Returns:
        Tensor, one-hot tensor.

    Raises:
        TypeError: If `num_classes` is not an int.
        TypeError: If dtype of `tensor` is not int32 or int64.
        ValueError: If `num_classes` is less than -1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> tensor = Tensor(np.array([0, 1, 2]), mindspore.int32)
        >>> num_classes = 3
        >>> output = mint.nn.functional.one_hot(tensor, num_classes)
        >>> print(output)
        [[1 0 0]
         [0 1 0]
         [0 0 1]]
    """
    return ops.function.array_func.one_hot_ext(tensor, num_classes)


def smooth_l1_loss(input, target, reduction='mean', beta=1.0):
    r"""
    Computes smooth L1 loss, a robust L1 loss.

    SmoothL1Loss is a Loss similar to MSELoss but less sensitive to outliers as described in the
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_ by Ross Girshick.

    Given two inputs :math:`x,\  y` of length :math:`N`, the SmoothL1Loss can be described
    as follows:

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\beta}, & \text{if } |x_i - y_i| < \beta \\
        |x_i - y_i| - 0.5 * \beta, & \text{otherwise. }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        L =
        \begin{cases}
            \operatorname{mean}(L_{i}), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L_{i}),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    Here :math:`\text{beta}` controls the point where the loss function changes from quadratic to linear.
    :math:`\text{beta} \geq 0` , its default value is ``1.0`` . :math:`N` is the batch size.

    .. warning::
        This is an experimental optimizer API that is subject to change.

    Note:
        - Arg `input` and `target` comply with the implicit type conversion rules to make the data types consistent.
          If they have different data types, the lower precision data type will be converted to relatively the
          highest precision data type.

    Args:
        input (Tensor): Tensor of shape :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
            Supported dtypes:
                Ascend: float16, float32, bfloat16.
        target (Tensor): Ground truth data, tensor of shape :math:`(N, *)`, same shape as the `input`.
            Supported dtypes:
                Ascend: float16, float32, bfloat16.
        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute the mean of elements in the output.
            - ``'sum'``: the output elements will be summed.
        beta (number, optional): A parameter used to control the point where the function will change between
            L1 to L2 loss. The value should be greater than or equal to zero. Default: ``1.0`` .

    Returns:
        Tensor, the data type is the same as `input`.
        If `reduction` is ``'none'``, then output is a tensor with the same shape as `input`.
        Otherwise, the shape of output tensor is :math:`()`.

    Raises:
        TypeError: If `input` or `target` is not a Tensor.
        RuntimeError: If dtype of `input` or `target` is not one of float16, float32, bfloat16.
        ValueError: If shape of `input` is not the same as `target`.
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'``, ``'sum'``.
        RuntimeError: If `beta` is not a float, int or bool.
        RuntimeError: If `beta` is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([2, 2, 3]), mindspore.float32)
        >>> target = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> beta = 1.0
        >>> reduction_1 = 'none'
        >>> output = ops.nn.functional.smooth_l1_loss(input, target, reduction_1, beta)
        >>> print(output)
        [0.  0.  0.5]
        >>> reduction_2 = 'mean'
        >>> output = ops.nn.functional.smooth_l1_loss(input, target, reduction_2, beta)
        >>> print(output)
        0.16666667
        >>> reduction_3 = 'sum'
        >>> output = ops.nn.functional.smooth_l1_loss(input, target, reduction_3, beta)
        >>> print(output)
        0.5
    """
    return ops.function.smooth_l1_loss(input, target, beta, reduction)


@constexpr
def log_warning(msg):
    """Adds warning to logger."""
    logger.warning(msg)


def dropout2d(input, p=0.5, training=True):
    r"""
    During training, randomly zeroes some channels of the input tensor with probability `p`
    from a Bernoulli distribution(For a 4-dimensional tensor with a shape of :math:`NCHW`,
    the channel feature map refers to a 2-dimensional feature map with the shape of :math:`HW`).

    For example, the :math:`j\_th` channel of the :math:`i\_th` sample in the batched input is a to-be-processed
    `2D` tensor input[i,j].
    Each channel will be zeroed out independently on every forward call which based on Bernoulli distribution
    probability `p`.
    The parper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ mentioned this technology, and it is proved that
    it can effectively reduce over fitting and prevent neuronal coadaptation.
    For more details, refer to `Improving neural networks by preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_ .

    `dropout2d` can improve the independence between channel feature maps.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): A `4D` tensor with shape :math:`(N, C, H, W)`, where `N` is the batch size, `C` is the number
            of channels, `H` is the feature height, and `W` is the feature width.
        p (float): The dropping probability of a channel, between 0 and 1, e.g. `p` = 0.8,
            which means dropping out 80% of channels. Default: ``0.5`` .
        training(bool): If `training` is True, applying dropout, otherwise, not applying. Default: ``True`` .

    Returns:
        Tensor, output, with the same shape and data type as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If the data type of `p` is not float.
        ValueError: If `p` is out of the range `[0.0, 1.0]`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.ones([2, 1, 2, 3]), mindspore.float32)
        >>> output = mint.nn.functional.dropout2d(input, 0.5)
        >>> print(output.shape)
        (2, 1, 2, 3)
    """
    def dropout2d_impl_(input, p, training):
        if p == 0 or not training or input.numel() == 0:
            return input

        if p == 1:
            return mint.mul(input, mint.zeros((), dtype=input.dtype))

        if input.ndim < 2:
            raise ValueError(f'For dropout2d, input size after unsqueeze must be greater or equal to 2')

        if ops.is_sequence_shape_unknown(input.shape):
            input_tensor_shape = ops.TensorShape()(input)
            nosie_tensor_shape = mint.ones_like(input_tensor_shape)
            nosie_tensor_shape[0] = input_tensor_shape[0]
            nosie_tensor_shape[1] = input_tensor_shape[1]
            nosie_shape = ops.TensorToTuple()(nosie_tensor_shape)
        else:
            nosie_shape = input.shape[:2] + tuple(1 for _ in range(len(input.shape) - 2))
        nosie = mint.full(nosie_shape, 1 - p, dtype=input.dtype)
        nosie = mint.bernoulli(nosie)
        nosie = mint.div(nosie, 1 - p)

        return mint.mul(input, nosie)

    validator.check_float_range(p, 0.0, 1.0, validator.INC_BOTH, "p", "dropout2d")
    validator.check_bool(training, "training", "dropout2d")

    if input.ndim not in (3, 4):
        log_warning(f"dropout2d receviced a {input.ndim}-D input which is not recommended. Please use dropout instead.")

    is_batched = input.ndim == 4
    if not is_batched:
        input_shape = input.shape
        if ops.is_sequence_shape_unknown(input.shape):
            input_shape = ops.TensorToTuple()(ops.TensorShape()(input))
        input = input.reshape((1, *input_shape))
        result = dropout2d_impl_(input, p, training)
        result = result.reshape(input_shape)
    else:
        result = dropout2d_impl_(input, p, training)

    return result

def normalize(input, p=2.0, dim=1, eps=1e-12):
    r"""
    Perform normalization of inputs over specified dimension

    For a tensor input of sizes :math:`(n_{0},..., n_{dim},..., n_{k})`, each :math:`n_{dim}` -element vector `v`
    along dimension `dim` is transformed as

    .. math::
        v=\frac{v}{\max(\left \| v \right \| _{p},\in )}

    With the default arguments it uses the Euclidean norm over vectors along dimension ``1`` for normalization.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): input tensor of any shape.
        p (float): the exponent value in the norm formulation. default: ``2``.
        dim (int): the dimension to reduce. default: ``1``.
        eps (float): small value to avoid division by zero. default: ``1e-12``.

    Returns:
        Tensor, shape and data type are the same as input.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> tensor = Tensor(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), mindspore.float32)
        >>> output = mint.nn.functional.normalize(tensor)
        >>> print(output)
        [[0.0000 0.4472 0.8944]
         [0.4243 0.5657 0.7071]
         [0.4915 0.5735 0.6554]]
    """
    denom = broadcast_to(clamp(norm_ext(input, p, dim, keepdim=True), min=eps), input.shape)
    return input / denom



__all__ = [
    'conv_transpose2d',
    'max_pool2d',
    # 1
    'binary_cross_entropy_with_logits',
    # 2

    # 3

    # 4
    "interpolate",
    # 5
    'pad',
    # 6
    'unfold',
    # 7
    'fold',
    # 8
    'layer_norm',
    # 9

    # 10

    # 11
    'relu',

    'relu_',

    # 12

    # 13

    # 14
    'dropout',
    # 15
    'conv2d',
    # 16
    'log_softmax',
    # 17

    # 18
    'prelu',
    # 19
    'binary_cross_entropy',
    # 20
    'cross_entropy',
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
    'softmax',
    # 32

    # 33

    # 34
    'batch_norm',
    # 35

    # 36
    'gelu',
    # 37

    # 38
    'linear',
    # 39
    'group_norm',
    # 40

    # 41

    # 42

    # 43

    # 44

    # 45

    # 46
    'silu',
    # 47

    # 48

    # 49
    'sigmoid',
    # 50

    # 51

    # 52
    'embedding',
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
    'one_hot',
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
    'avg_pool2d',
    # 91
    'grid_sample',
    # 92
    'leaky_relu',
    # 93

    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100

    # 312
    'normalize',

    # 323

    # 324
    'elu',
    # 325

    #556
    'logsigmoid',

    # 257

    # 258
    'mse_loss',
    # 259

    'adaptive_avg_pool1d',

    'adaptive_avg_pool2d',
    # 393
    'dropout2d',
]
