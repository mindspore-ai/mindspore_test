mindspore.ops.erfinv
====================

.. py:function:: mindspore.ops.erfinv(input)

    逐元素计算输入tensor的逆误差。
    逆误差函数在 `(-1, 1)` 范围内定义为：

    .. math::
        erfinv(erf(x)) = x

    参数：
        - **input** (Tensor) - 输入tensor。
    返回：
        Tensor
