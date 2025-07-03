mindspore.ops.ne
========================

.. py:function:: mindspore.ops.ne(input, other)

    逐元素计算两个输入是否不相等。

    .. math::
        out_{i} =\begin{cases}
        & \text{True,    if } input_{i} \ne other_{i} \\
        & \text{False,   if } input_{i} = other_{i}
        \end{cases}

    .. note::
        - 支持隐式类型转换。
        - 当输入是两个Tensor时，它们的shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入。

    返回：
        Tensor
