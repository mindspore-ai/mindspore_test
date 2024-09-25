mindspore.numpy.floor_divide
============================

.. py:function:: mindspore.numpy.floor_divide(x1, x2, dtype=None)

    对输入进行除法运算，返回小于等于除法结果的最大整数。 该函数等同于Python中的 // 运算符，并与运算符 % (取余)配对使用，使得 a = a % b + b * (a // b) 在四舍五入后成立。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入数组。
        - **x2** (Tensor) - 输入数组。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。
