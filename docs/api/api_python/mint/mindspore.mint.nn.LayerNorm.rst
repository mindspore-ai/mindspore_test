mindspore.mint.nn.LayerNorm
===========================

.. py:class:: mindspore.mint.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, dtype=None)

    在mini-batch输入上应用层归一化（Layer Normalization）。

    层归一化在递归神经网络中被广泛的应用。适用单个训练用例的mini-batch输入上应用归一化，详见论文 `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_ 。

    与批归一化（Batch Normalization）不同，层归一化在训练和测试时执行完全相同的计算。
    应用于所有通道和像素，即使batch_size=1也适用。其中 :math:`\gamma` 是通过训练学习出的scale值，:math:`\beta` 是通过训练学习出的shift值。公式如下：

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **normalized_shape** (Union(tuple[int], list[int], int)) - 对 `x` 执行归一化的shape。
        - **eps** (float，可选) - 添加到分母中的值（:math:`\epsilon`），以确保数值稳定。默认值： ``1e-5`` 。
        - **elementwise_affine** (bool，可选) - 是否需要仿射变换。当被设置为 ``True`` 时，权重参数初始化为1，偏差初始化为0。默认值： ``True`` 。
        - **bias** (bool，可选) - 当被设置为 ``False`` 时，不会学习结果性偏差（仅 `elementwise_affine` 值为 ``True`` 时生效）。默认值： ``True`` 。
        - **dtype** (:class:`mindspore.dtype`，可选) - Parameters的dtype。默认值： ``None`` 。

    输入：
        - **x** (Tensor) - `x` 的shape为 :math:`(N, *)` ， `*` 等于 `normalized_shape` 。

    输出：
        Tensor，归一化后的Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `eps` 不是float。