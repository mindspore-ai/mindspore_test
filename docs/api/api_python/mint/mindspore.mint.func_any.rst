mindspore.mint.any
=====================

.. py:function:: mindspore.mint.any(input, dim=None, keepdim=False)

    检查指定维度上是否含有 `True`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, tuple(int), list(int), Tensor], 可选) - 要减少的维度。如果为 ``None``，减少所有维度。默认 ``None`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
