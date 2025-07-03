mindspore.ops.cos
==================

.. py:function:: mindspore.ops.cos(input)

    逐元素计算输入tensor的余弦。

    .. math::
        out_i = \cos(x_i)

    .. warning::
        如果使用float64，可能会存在精度丢失的问题。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor