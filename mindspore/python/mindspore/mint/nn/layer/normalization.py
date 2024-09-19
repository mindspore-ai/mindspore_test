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

from typing import Optional
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore import ops
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.nn.layer.normalization import LayerNormExt as LayerNorm
from mindspore.ops import group_norm


class _NormBase(Cell):
    """Common base of _InstanceNorm and _BatchNorm"""

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 dtype=None
                 ) -> None:
        super(_NormBase, self).__init__()
        self.shape = ops.Shape()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.dtype = dtype if dtype is not None else mstype.float32
        if self.affine:
            self.weight = Parameter(
                Tensor(np.empty(num_features), dtype=self.dtype), name="weight")
            self.bias = Parameter(
                Tensor(np.empty(num_features), dtype=self.dtype), name="bias")
            self.weight: Optional[Parameter]
            self.bias: Optional[Parameter]
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            self.running_mean = Parameter(Tensor(np.zeros(num_features), dtype=self.dtype),
                                          requires_grad=False, name="running_mean")
            self.running_var = Parameter(Tensor(np.ones(num_features), dtype=self.dtype),
                                         requires_grad=False, name="running_var")
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.num_batches_tracked = Parameter(Tensor(0, dtype=ms.float32),
                                                 requires_grad=False, name="num_batches_tracked")
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        """init parameters"""

        if self.track_running_stats:
            zero_running_mean = Tensor(
                np.zeros(self.num_features), dtype=self.dtype)
            one_running_var = Tensor(
                np.ones(self.num_features), dtype=self.dtype)
            zero_num_batches_tracked = Tensor(0, dtype=ms.float32)

            ops.assign(self.running_mean, zero_running_mean)
            ops.assign(self.running_var, one_running_var)
            ops.assign(self.num_batches_tracked, zero_num_batches_tracked)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            one_weight = Tensor(np.ones(self.num_features), dtype=self.dtype)
            zero_bias = Tensor(np.zeros(self.num_features), dtype=self.dtype)

            ops.assign(self.weight, one_weight)
            ops.assign(self.bias, zero_bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, affine={}, track_running_stats={}'.format(
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)


class _BatchNorm(_NormBase):
    """common base of BatchNormXxx"""

    def __init__(
            self,
            num_features: int,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            dtype=None) -> None:
        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats,
                                         dtype)
        self.training = True

    def _check_input_dim(self, input):
        raise NotImplementedError

    def construct(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                num_batches_tracked_one = Tensor(1, dtype=ms.float32)
                ops.assign_add(self.num_batches_tracked,
                               num_batches_tracked_one)
                if self.momentum is None:
                    exponential_average_factor = float(1.0 / self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        return mint.functional.batch_norm(
            input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats
            else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm1d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 2D or 3D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ .

    .. math::

        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.

    .. warning::
        This API does not support Dynamic Rank.
        This is an experimental API that is subject to change or deletion.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, L)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``True`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C)` or :math:`(N, C, L)`,
          where :math:`N` means batch, :math:`C` means the number of feature or the number of channel,
          and :math:`L` is the length of sequence.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        TypeError: If `num_features` is not a int number.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> input_x = mindspore.Tensor([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]])
        >>> net = mint.nn.BatchNorm1d(4)
        >>> output = net(input_x)
        >>> print(output)
        [[ 0.99950075 0.9980011 -0.9980068 -0.9997783]
         [-0.9995012 -0.99799967 0.9980068  0.9997778]]
    """

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim != 2 and dim != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(dim)
            )


class BatchNorm2d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 4D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ .

    .. math::

        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.

    .. warning::
        This API does not support Dynamic Rank.
        This is an experimental API that is subject to change or deletion.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, H, W)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``True`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, H, W)`.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        TypeError: If `num_features` is not a int number.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> input_x = mindspore.Tensor([0.3, 0.4, 0.5, 0.3])
        >>> input_x = input_x.reshape((2, 2, 1, 1))
        >>> net = mint.nn.BatchNorm2d(2)
        >>> output = net(input_x)
        >>> print(output)
        [[[[-0.99950075]]
          [[0.9980087]]]
          [[[0.999501]]
          [[-0.9980097]]]]
    """

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim != 4:
            raise ValueError(
                "expected 4D input (got {}D input)".format(dim)
            )


class BatchNorm3d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 5D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ .

    .. math::

        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.

    .. warning::
        This API does not support Dynamic Rank.
        This is an experimental API that is subject to change or deletion.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, D, H, W)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``True`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, D, H, W)`.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        TypeError: If `num_features` is not a int number.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> input_x = mindspore.Tensor([0.1, 0.9, 1.2, 2.3])
        >>> input_x = input_x.reshape((1, 2, 1, 1, 2))
        >>> net = mint.nn.BatchNorm3d(2)
        >>> output = net(input_x)
        >>> print(output)
        [[[[[-0.9999688 0.99996865]]]
          [[[-0.9999833 06.9999831]]]]]
    """

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim != 5:
            raise ValueError(
                "expected 5D input (got {}D input)".format(dim)
            )


class GroupNorm(Cell):
    r"""
    Group Normalization over a mini-batch of inputs.

    Group Normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_.

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization, and it performs very stable over a wide
    range of batch size. :math:`\gamma` and :math:`\beta` are trainable scale and shift.
    It can be described using the following formula:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is `weight`, :math:`\beta` is `bias`, and :math:`\epsilon` is `eps`.

    Args:
        num_groups (int): The number of groups to be divided along the channel dimension.
        num_channels (int): The number of input channels.
        eps (float, optional): A value added to the denominator for numerical stability. Default: ``1e-05`` .
        affine (bool, optional): The parameters, such as :math:`\gamma` and :math:`\beta`, are learnable
            when set to ``true`` . Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input feature with shape :math:`(N, C, *)`, where :math:`*` means, any number of
          additional dimensions.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `x`.

    Raises:
        TypeError: If `num_groups` or `num_channels` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If `affine` is not a bool.
        ValueError: If `num_groups` or `num_channels` is less than 1.
        ValueError: If `num_channels` is not divided by `num_groups`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> group_norm_op = ms.mint.nn.GroupNorm(2, 2)
        >>> x = ms.Tensor(np.ones([1, 2, 4, 4], np.float32))
        >>> output = group_norm_op(x)
        >>> print(output)
        [[[[0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]]
          [[0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]]]]
    """

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, dtype=None):
        """Initialize GroupNorm."""
        super(GroupNorm, self).__init__()
        ms_dtype = mstype.float32 if dtype is None else dtype
        gamma_init = 'ones'
        beta_init = 'zeros'

        self.num_groups = validator.check_positive_int(
            num_groups, "num_groups", self.cls_name)
        self.num_channels = validator.check_positive_int(
            num_channels, "num_channels", self.cls_name)
        if num_channels % num_groups != 0:
            raise ValueError(f"For '{self.cls_name}', the 'num_channels' must be divided by 'num_groups', "
                             f"but got 'num_channels': {num_channels}, 'num_groups': {num_groups}.")
        self.eps = validator.check_value_type(
            'eps', eps, (float,), type(self).__name__)
        self.affine = validator.check_bool(
            affine, arg_name="affine", prim_name=self.cls_name)

        self.gamma = Parameter(initializer(
            gamma_init, self.num_channels, dtype=ms_dtype), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, self.num_channels, dtype=ms_dtype), name="beta", requires_grad=affine)

    def _cal_output(self, x):
        """calculate groupnorm output"""
        return group_norm(x, self.num_groups, self.gamma, self.beta, self.eps)

    def extend_repr(self):
        return 'num_groups={}, num_channels={}, eps={}, affine={}'.format(
            self.num_groups, self.num_channels, self.eps, self.affine)

    def construct(self, input):
        output = self._cal_output(input)
        return output


__all__ = [
    'GroupNorm',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
]
