mindspore.mint.nn.AdaptiveAvgPool1d
====================================

.. py:class:: mindspore.mint.nn.AdaptiveAvgPool1d(output_size)

    对由多个输入平面组成的输入信号应用1D自适应平均池化。

    对于任何输入大小，输出大小为 :math:`L_{out}` 。
    输出特征的数量等于输入平面的数量。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **output_size** - 目标输出的size :math:`L_{out}` 。

    输入：
        - **input** (Tensor) - 输入特征的shape为 :math:`(N, C, L_{in})` 或  :math:`(C, L_{in})` 。
