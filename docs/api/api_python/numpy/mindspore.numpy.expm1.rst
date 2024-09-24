mindspore.numpy.expm1
=====================

.. py:function:: mindspore.numpy.expm1(x, dtype=None)

    计算数组中所有元素的 ``exp(x) - 1``  。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。
        在GPU上，支持的数据类型有np.float16和np.float32。在CPU上，支持的数据类型有np.float16，np.float32和np.float64。

    参数：
        - **x** (Tensor) - 输入数据。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。 逐元素计算 ``exp(x) - 1`` 。 如果 `x1` 和 `x2` 都是标量，返回标量。