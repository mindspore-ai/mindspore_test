mindspore.ops.log
=================

.. py:function:: mindspore.ops.log(input)

    逐元素计算输入tensor的自然对数。

    .. math::
        y_i = \log_e(x_i)

    .. warning::
        如果输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
