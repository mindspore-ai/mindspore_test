mindspore.ops.any
=================

.. py:function:: mindspore.ops.any(input, axis=None, keep_dims=False)

    检查指定维度上是否含有 `True`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 要减少的维度。如果为 ``None``，减少所有维度。
        - **keep_dims** (bool, 可选) - 输出tensor是否保留维度。

    返回：
        Tensor