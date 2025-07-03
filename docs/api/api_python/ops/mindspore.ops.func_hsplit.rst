mindspore.ops.hsplit
=====================

.. py:function:: mindspore.ops.hsplit(input, indices_or_sections)

    将输入tensor水平分割成多个子tensor。等同于 :math:`axis=1` 时的 `ops.tensor_split` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 参考 :func:`mindspore.ops.tensor_split`。

    返回：
        由多个tensor组成的tuple。
