mindspore.ops.nanmedian
=======================

.. py:function:: mindspore.ops.nanmedian(input, axis=-1, keepdims=False)

    忽略NaN值，按指定轴计算输入tensor的中值和索引。

    .. warning::
        如果 `input` 的中值不唯一，则 `indices` 不一定包含第一个出现的中值。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int, 可选) - 指定轴，默认 ``-1`` 。
        - **keepdims** (bool, 可选) - 输出tensor是否保持维度，默认 ``False`` 。

    返回：
        一个由tensor组成的tuple(median, median_indices)
