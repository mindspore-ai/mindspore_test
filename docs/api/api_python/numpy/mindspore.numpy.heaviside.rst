mindspore.numpy.heaviside
=========================

.. py:function:: mindspore.numpy.heaviside(x1, x2, dtype=None)

    计算Heaviside阶跃函数。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入值。
        - **x2** (Tensor) - 当 `x1` 为0时函数的值。 如果 ``x1.shape != x2.shape`` ，他们必须能广播到一个共同的shape(即输出的shape)。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，输出数组，逐元素计算 `x1` 的Heaviside阶跃函数。 如果 `x1` 和 `x2` 都是标量，返回标量。