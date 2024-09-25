mindspore.numpy.cos
===================

.. py:function:: mindspore.numpy.cos(x, dtype=None)

    逐元素计算余弦。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype 。

    返回：
        Tensor或标量。 如果 `x` 是标量，则结果也是标量。