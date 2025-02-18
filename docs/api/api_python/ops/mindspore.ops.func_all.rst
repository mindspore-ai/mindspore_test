mindspore.ops.all
=================

.. py:function:: mindspore.ops.all(input, axis=None, keep_dims=False)

    检查指定维度上是否所有元素均为True。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 要减少的维度。默认值： ``None`` ，减少所有维度。
        - **keep_dims** (bool, 可选) - 输出Tensor是否保留维度，默认值： ``False`` 。

    返回：
        Tensor