mindspore.ops.all
=================

.. py:function:: mindspore.ops.all(input, axis=None, keep_dims=False)

    检查指定轴上是否所有元素均为 `True`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 指定轴。如果为 ``None``，检查所有元素。默认 ``None`` 。
        - **keep_dims** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor