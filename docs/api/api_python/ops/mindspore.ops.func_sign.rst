mindspore.ops.sign
===================

.. py:function:: mindspore.ops.sign(input)

    按sign公式逐元素计算输入tensor。

    .. math::
        \text{out}_{i} = \begin{cases}
                          -1 & \text{input}_{i} < 0 \\
                           0 & \text{input}_{i} = 0 \\
                           1 & \text{input}_{i} > 0
                         \end{cases}

    .. note::
        在输入为NaN且数据类型为float64时，该算子计算结果为NaN。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
