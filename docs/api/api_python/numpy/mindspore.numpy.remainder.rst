mindspore.numpy.remainder
=========================

.. py:function:: mindspore.numpy.remainder(x1, x2, dtype=None)

    逐元素返回除法余数。

    计算与 `floor_divide` 函数互补的余数。相当于 Python 的取模运算符 ``x1 % x2`` ，并且具有与除数 `x2` 相同的符号。与 `np.remainder` 等效的 MATLAB 函数是 mod 。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入数组。
        - **x2** (Tensor) - 输入数组。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor 或标量，逐元素计算 ``floor_divide(x1, x2)`` 得到的余数。如果 `x1` 和 `x2` 都是标量，则返回标量。