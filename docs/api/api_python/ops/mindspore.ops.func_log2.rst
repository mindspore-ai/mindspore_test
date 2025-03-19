mindspore.ops.log2
===================

.. py:function:: mindspore.ops.log2(input)

    逐元素计算输入tensor以2为底的对数。

    .. math::
        y_i = \log_2(input_i)

    .. warning::
        如果log2的输入值范围在(0, 0.01]或[0.95, 1.05]区间，输出精度可能会受影响。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
