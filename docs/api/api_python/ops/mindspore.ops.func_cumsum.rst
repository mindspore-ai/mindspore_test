mindspore.ops.cumsum
====================

.. py:function:: mindspore.ops.cumsum(x, axis, dtype=None)

    返回tensor在指定轴上累积的元素和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ... + x_i

    参数：
        - **x** (Tensor) - 输入tensor。
        - **axis** (int) - 指定计算的轴。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型。默认 ``None`` 。
    返回：
        Tensor
