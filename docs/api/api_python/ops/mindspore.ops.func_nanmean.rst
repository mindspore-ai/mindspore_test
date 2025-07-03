mindspore.ops.nanmean
=====================

.. py:function:: mindspore.ops.nanmean(input, axis=None, keepdims=False, *, dtype=None)

    忽略NaN值，按指定轴计算输入tensor的平均值。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int, 可选) - 指定轴，默认 ``None`` 。
        - **keepdims** (bool, 可选) - 输出tensor是否保留维度，默认 ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型，默认 ``None`` 。

    返回：
        Tensor
