mindspore.ops.channel_shuffle
=============================

.. py:function:: mindspore.ops.channel_shuffle(x, groups)

    将shape为 :math:`(*, C, H, W)` 的tensor的通道划分成 :math:`g` 组，并按如下方式重新排列 :math:`(*, \frac{C}{g}, g, H*W)` ，同时在最终输出中保持原始tensor的shape。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **groups** (int) - 通道划分数目。

    返回：
        Tensor
