mindspore.numpy.true_divide
===========================

.. py:function:: mindspore.numpy.true_divide(x1, x2, dtype=None)

    返回输入的真除法，逐元素计算。

    该函数与Python默认的“向下取整除”不同，它返回真除法的结果。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 被除数。
        - **x2** (Tensor) - 除数。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，如果 `x1` 和 `x2` 都是标量，则返回标量。