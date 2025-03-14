# Copyright 2025 Huawei Technologies Co., Ltd
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
"""XInstanceNormX for mint"""

from __future__ import absolute_import
from __future__ import division

from mindspore.ops.function.nn_func import instance_norm
from .normalization import _NormBase


class _InstanceNorm(_NormBase):
    """common base of InstanceNormXXX"""

    def __init__(
            self,
            num_features: int,
            eps=1e-5,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
            device=None,
            dtype=None) -> None:
        super(_InstanceNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats,
                                            dtype)
        self.training = True
        if device is not None:
            raise ValueError(f"expected device is None value, but got {device}.")

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _handle_no_batch_input(self, input):
        return self._apply_instance_norm(input.unsqueeze(0)).squeeze(0)

    def _apply_instance_norm(self, input):
        return instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                             self.training or not self.track_running_stats, self.momentum, self.eps)

    def construct(self, input):
        self._check_input_dim(input)

        input_shape = self.shape(input)
        feature_dim = len(input_shape) - self._get_no_batch_dim()
        if input_shape[feature_dim] != self.num_features:
            if self.affine:
                raise ValueError(f"expected input's size at dim={feature_dim} to match num_features"
                                 f" ({self.num_features}), but got: {input_shape[feature_dim]}.")

        if len(input_shape) == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)
        return self._apply_instance_norm(input)


class InstanceNorm1d(_InstanceNorm):
    r"""
    This layer applies Instance Normalization over a 3D input (a mini-batch of 1D inputs with
    additional channel dimension). Refer to the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for each object in a mini-batch.
    :math:`\gamma` and :math:`\beta` are learnable parameter vectors of size `num_features` if `affine` is ``True``.
    The standard-deviation is calculated via the biased estimator.

    This layer uses instance statistics computed from input data in both training and evaluation modes.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, L)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the `running_mean` and `running_var`
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``False`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``False`` .
        device (None, optional): Only supports None values. Default: ``None`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, L)` or :math:`(C, L)`.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        TypeError: If `num_features` is not an int number.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input_x = Tensor(np.random.randn(2, 4, 8))
        >>> net = mint.nn.InstanceNorm1d(4)
        >>> output = net(input_x)
        >>> out.shape
        (2, 4, 8)
    """

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim not in (2, 3):
            raise ValueError(
                "expected 2D or 3D input (got {}D input).".format(dim)
            )


class InstanceNorm2d(_InstanceNorm):
    r"""
    This layer applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with
    additional channel dimension). Refer to the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for each object in a mini-batch.
    :math:`\gamma` and :math:`\beta` are learnable parameter vectors of size `num_features` if `affine` is ``True``.
    The standard-deviation is calculated via the biased estimator.

    This layer uses instance statistics computed from input data in both training and evaluation modes.

    InstanceNorm2d and BatchNorm2d are very similar, but have some differences. InstanceNorm2d is applied on each
    channel of channeled data like RGB images, but BatchNorm2d is usually applied on each batch of batched data.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, H, W)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``False`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``False`` .
        device (None, optional): Only supports None values. Default: ``None`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        TypeError: If `num_features` is not an int number.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input_x = Tensor(np.random.randn(2, 4, 8, 8))
        >>> net = mint.nn.InstanceNorm2d(4)
        >>> output = net(input_x)
        >>> out.shape
        (2, 4, 8, 8)
    """

    def _get_no_batch_dim(self):
        return 3

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim not in (3, 4):
            raise ValueError(
                "expected 3D or 4D input (got {}D input).".format(dim)
            )


class InstanceNorm3d(_InstanceNorm):
    r"""
    This layer applies Instance Normalization over a 5D input (a mini-batch of 3D inputs with
    additional channel dimension). Refer to the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for each object in a mini-batch.
    :math:`\gamma` and :math:`\beta` are learnable parameter vectors of size `num_features` if `affine` is ``True``.
    The standard-deviation is calculated via the biased estimator.

    This layer uses instance statistics computed from input data in both training and evaluation modes.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, D, H, W)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``False`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``False`` .
        device (None, optional): Only supports None values. Default: ``None`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        TypeError: If `num_features` is not an int number.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input_x = Tensor(np.random.randn(2, 4, 8, 8, 8))
        >>> net = mint.nn.InstanceNorm3d(4)
        >>> output = net(input_x)
        >>> out.shape
        (2, 4, 8, 8, 8)
    """

    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim not in (4, 5):
            raise ValueError(
                "expected 4D or 5D input (got {}D input).".format(dim)
            )


__all__ = [
    'InstanceNorm1d',
    'InstanceNorm2d',
    'InstanceNorm3d',
]
