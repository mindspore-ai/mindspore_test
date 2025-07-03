mindspore.ops.vsplit
=====================

.. py:function:: mindspore.ops.vsplit(input, indices_or_sections)

    根据 `indices_or_sections` 将至少有两维的输入tensor垂直分割成多个子tensor。

    等同于 :math:`axis=0` 时的 `ops.tensor_split` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 参考 :func:`mindspore.ops.tensor_split` 中的 `indices_or_sections` 参数。

    返回：
        由多个tensor组成的tuple。
