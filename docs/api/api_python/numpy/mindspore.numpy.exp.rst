mindspore.numpy.exp
===================

.. py:function:: mindspore.numpy.exp(x, dtype=None)

    计算输入数组中所有元素的指数。

    .. note::
        不支持NumPy参数 `casting` 、 `order` 、 `subok` 、 `signature` 和 `extobj` 。
        当使用 `where` 时，`out` 必须Tensor值。 `out` 不支持用于存储结果，但可以与 `where` 一起使用来设置指定索引的值。
        在GPU上，支持的数据类型有np.float16和np.float32。在CPU上，支持的数据类型有np.float16，np.float32和np.float64。

    参数：
        - **x** (Tensor) - 输入数据。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，逐元素计算的指数值。 如果 `x1` 和 `x2` 都是标量，返回标量。