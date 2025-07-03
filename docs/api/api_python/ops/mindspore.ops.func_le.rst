mindspore.ops.le
========================

.. py:function:: mindspore.ops.le(input, other)

    逐元素计算 :math:`input <= other` 的值。

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } input_{i}<=other_{i} \\
            & \text{False,   if } input_{i}>other_{i}
            \end{cases}

    .. note::
        - 支持隐式类型转换。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入。

    返回：
        Tensor
