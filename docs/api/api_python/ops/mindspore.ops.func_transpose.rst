mindspore.ops.transpose
=======================

.. py:function:: mindspore.ops.transpose(input, input_perm)

    根据指定的维度排列顺序对输入tensor进行维度转置。

    .. note::
        GPU和CPU平台上，如果 `input_perm` 的元素值为负数，则其实际值为 `input_perm[i] + rank(input)` 。 Ascend平台不支持 `input_perm` 元素值为负。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **input_perm** (tuple[int]) - 指定轴的新排列。

    返回：
        Tensor