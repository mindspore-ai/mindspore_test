mindspore.ops.movedim
======================

.. py:function:: mindspore.ops.movedim(x, source, destination)

    将输入tensor的两个维度调换位置。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **source** (Union[int, sequence[int]]) - 源维度。
        - **destination** (Union[int, sequence[int]]) - 源维度的目标位置。

    返回：
        Tensor
