mindspore.mint.nn.InstanceNorm2d
================================

.. py:class:: mindspore.mint.nn.InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)

    该层对4D输入（具有额外通道维度的2D输入的小批量数据）应用实例归一化（Instance Normalization）。
    具体内容可参考论文 `Instance Normalization: The Missing Ingredient for Fast Stylization <https://arxiv.org/abs/1607.08022>`_ 。
    它使用小批量数据和可学习的参数对特征进行重新缩放和重新中心化，公式如下：

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    均值和标准差是针对小批量数据中每个对象的每个维度分别计算的。如果 `affine` 为 ``True`` ，则 :math:`\gamma` 和 :math:`\beta` 是大小为 `num_features` 的可学习参数向量。标准差通过有偏估计计算。
    
    该层在训练和评估模式下都使用从输入数据计算的实例统计量。

    InstanceNorm2d 和 BatchNorm2d 非常相似，但有一些区别。InstanceNorm2d应用于通道数据（如RGB图像）的每个通道，而BatchNorm2d通常应用于批量数据的每个批次。
    
    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **num_features** (int) - 输入的shape :math:`(N, C, H, W)` 中的 `C` 值。
        - **eps** (float, 可选) - 添加到分母中的值，用于数值稳定性。默认值： ``1e-5`` 。
        - **momentum** (float, 可选) - 用于计算 `running_mean` 和 `running_var` 的值。可以设置为 ``None`` 以使用累积移动平均。默认值： ``0.1`` 。
        - **affine** (bool, 可选) - bool类型。设置为 ``True`` 时，可学习到反射参数值。默认值： ``False`` 。
        - **track_running_stats** (bool, 可选) - bool类型。设置为 ``True`` 时，会跟踪运行时的均值和方差。当设置为 ``False`` 时，
          则不会跟踪这些统计信息。且在train和eval模式下，该cell总是使用batch的统计信息。默认值： ``False`` 。
        - **device** (None, 可选) - 只支持None。默认值： ``None`` 。
        - **dtype** (:class:`mindspore.dtype`, 可选) - Parameters的dtype。默认值： ``None`` 。

    输入：
        - **input** (Tensor) - 输入shape为 :math:`(N, C, H, W)` 或 :math:`(C, H, W)`。

    输出：
        Tensor，类型和shape与 `input` 一致。

    异常：
        - **TypeError** - `num_features` 不是整数。
        - **TypeError** - `eps` 不是浮点数。
        - **ValueError** - `num_features` 小于1。
