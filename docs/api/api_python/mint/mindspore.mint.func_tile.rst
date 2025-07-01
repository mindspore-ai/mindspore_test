mindspore.mint.tile
===================

.. py:function:: mindspore.mint.tile(input, dims)

    通过复制 `dims` 次输入tensor中的元素来创建新tensor。

    返回tensor的第i维度有 `input.shape[i] * dims[i]` 个元素，并且 `input` 的值沿第i维度被复制 `dims[i]` 次。

    .. note::
        - 在Ascend平台上， `dims` 参数的个数不大于8，当前不支持超过4个维度同时做被复制的场景。
        - 如果 `input.dim = d` ，将其相应位置的shape相乘，返回的shape为 :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)` 。
        - 如果 `input.dim < d` ，在 `input` 的shape的前面填充1，直到它们的长度一致。例如将 `input` 的shape设置为 :math:`(1, ..., x_1, x_2, ..., x_S)` ，然后可以将其相应位置的shape相乘，返回的shape为 :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)` 。
        - 如果 `input.dim > d` ，在 `dims` 的前面填充1，直到它们的长度一致。例如将 `dims` 设置为 :math:`(1, ..., y_1, y_2, ..., y_S)` ，然后可以将其相应位置的shape相乘，返回的shape为 :math:`(x_1*1, ..., x_R*y_R, x_S*y_S)` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dims** (tuple[int]) - 指定每维度的复制次数。

    返回：
        Tensor