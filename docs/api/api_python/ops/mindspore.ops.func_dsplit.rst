mindspore.ops.dsplit
=====================

.. py:function:: mindspore.ops.dsplit(input, indices_or_sections)

    沿第三轴分割输入tensor。等同于 :math:`axis=2` 时的 `ops.tensor_split` 。

    参数：
        - **input** (Tensor) - 待分割的tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 参考 :func:`mindspore.ops.tensor_split` 中的indices_or_sections参数。

    返回：
        多个tensor组成的tuple。