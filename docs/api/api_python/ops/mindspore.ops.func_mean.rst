mindspore.ops.mean
==================

.. py:function:: mindspore.ops.mean(x, axis=None, keep_dims=False)

    计算tensor在指定轴上的均值。

    参数：
        - **x** (Tensor[Number]) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor]) - 指定轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keep_dims** (bool) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
