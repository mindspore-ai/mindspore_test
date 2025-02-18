mindspore.mint.nn.ChannelShuffle
================================

.. py:class:: mindspore.mint.nn.ChannelShuffle(groups)

    将shape为 :math:`(*, C, H, W)` 的Tensor的通道划分成 :math:`g` 组，并按如下方式重新排列 :math:`(*, \frac{C}{g}, g, H*W)` ，同时在最终输出中保持原始Tensor的shape。

    .. note::
        C总是为输入的第二个维度。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **groups** (int) - 通道划分数目。

    输入：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(*, C, H, W)` 。

    输出：
        Tensor，数据类型与 `input` 相同。

    异常：
        - **TypeError** - `groups` 不是整数。
        - **ValueError** - `groups` 小于1。
        - **ValueError** - `input` 的维度小于3。
        - **ValueError** - `input` 的通道数不能被 `groups` 整除。