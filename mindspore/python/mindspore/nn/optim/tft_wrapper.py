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
"""OptTFTWrapper"""
from __future__ import absolute_import

import os
from mindspore.common.tensor import Tensor
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.ops.operations.manually_defined._inner import TensorReport
from mindspore import ops, context


class OptTFTWrapper(Optimizer):
    r"""
    Implements TFT optimizer wrapper, this wrapper is used to report status to MindIO TFT before optimizer updating.

    Note:
        This optimizer is depend on MindIO TFT feature. Currently only support ascend graph mode and
        sink_size must be less than 1.

    Args:
        opt (Optimizer): Must be sub-class of Optimizer.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of opt's `params`, the shape is the same as opt's `params`.

    Outputs:
        Tensor, result of executing optimizer 'opt'.

    Raises:
        TypeError: If the parameter opt is not an subclass of Optimizer.
        ValueError: If the platform is not Ascend graph mode, or customer doesn't switch on TFT feature.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.SGD(params=net.trainable_params())
        >>> optim_wrapper = nn.OptTFTWrapper(optim)
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.train.Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(self, opt):
        super(OptTFTWrapper, self).__init__(opt.learning_rate, opt._parameters) # pylint: disable=W0212
        if not isinstance(opt, Optimizer):
            raise TypeError(f"For 'OptTFTWrapper', the argument 'opt' must be Optimizer type, " f"but got {type(opt)}.")
        tft_env = os.getenv("MS_ENABLE_TFT", "")
        if ("TTP:1" not in tft_env) and ("UCE:1" not in tft_env):
            raise ValueError("MindIO TFT regitster need custom switch on[MS_ENABLE_TFT='{TTP:1,UCE:1}']!")
        mode = context.get_context("mode")
        device_target = context.get_context("device_target")
        if device_target != "Ascend" or mode != context.GRAPH_MODE:
            raise ValueError("MindIO adataper only support on Ascend device with GRAPH Mode!")
        self.opt = opt
        self.report = TensorReport()
        self.depend = ops.Depend()
        self.g_one = Tensor([0.1])

    def construct(self, gradients):
        g_one = self.depend(self.g_one, gradients)
        self.report("tft_report", g_one)
        return self.opt(gradients)
