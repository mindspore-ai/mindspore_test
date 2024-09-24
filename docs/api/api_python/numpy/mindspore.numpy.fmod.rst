mindspore.numpy.fmod
====================

.. py:function:: mindspore.numpy.fmod(x1, x2, dtype=None)

    返回除法的逐元素余数。

    这是C语言库中函数fmod的NumPy实现，余数与被除数 `x1` 的符号相同。 它等同于Matlab(TM)的rem函数，勿与Python的取模运算符 ``x1 % x2`` 混淆。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 第一个输入数组。
        - **x2** (Tensor) - 第二个输入数组。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。 `x1` 除以 `x2` 所得余数。 如果 `x1` 和 `x2` 都是标量，返回标量。