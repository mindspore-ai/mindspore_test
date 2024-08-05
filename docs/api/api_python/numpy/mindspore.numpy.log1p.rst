mindspore.numpy.log1p
=====================

.. py:function:: mindspore.numpy.log1p(x, dtype=None)

    返回1加上输入数组的自然对数，逐元素计算。

    计算 ``log(1 + x)`` 。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x** (Tensor) - 输入数组。
        - **dtype** (mindspore.dtype) - 默认值: `None` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。如果 `x` 是标量，返回标量。
