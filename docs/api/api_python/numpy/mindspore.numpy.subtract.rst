mindspore.numpy.subtract
========================

.. py:function:: mindspore.numpy.subtract(x1, x2, dtype=None)

    逐元素减去给定参数。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 被减数输入。
        - **x2** (Tensor) - 减数输入。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， `x1` 和 `x2` 的逐元素差。 如果 `x1` 和 `x2` 都是标量，则返回标量。