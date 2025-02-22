mindspore.mint.all
==================

.. py:function:: mindspore.mint.all(input)

    检查是否所有元素均为 `True`。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor

    .. py:function:: mindspore.mint.all(input, dim, keepdim=False)
        :noindex:

    检查指定维度上是否所有元素均为 `True`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, tuple(int), list(int), Tensor]) - 要减少的维度。如果为 ``None``，减少所有维度。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。

    返回：
        Tensor
