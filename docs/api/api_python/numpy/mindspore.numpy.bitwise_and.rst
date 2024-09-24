mindspore.numpy.bitwise_and
===========================

.. py:function:: mindspore.numpy.bitwise_and(x1, x2, dtype=None)

    逐元素计算两个数组的按位与运算。 计算输入数组中整数的二进制表示的按位与。 此函数实现了C/Python中的操作符 & 。
    
    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入数组。
        - **x2** (Tensor) - 输入数组。 只处理int和bool类型。 如果 ``x1.shape != x2.shape`` ，它们必须能广播到一个共同的shape(即输出的shape)。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。 若 `x` 是标量，则返回值也是标量。
