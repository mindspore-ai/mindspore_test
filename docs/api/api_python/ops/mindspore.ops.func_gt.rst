mindspore.ops.gt
=====================

.. py:function:: mindspore.ops.gt(input, other)

    逐元素计算 :math:`input > other` 的值。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } input_{i}>other_{i} \\
            & \text{False,   if } input_{i}<=other_{i}
            \end{cases}

    .. note::
        - 支持隐式类型转换。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是两个Tensor时，它们的数据类型不能同时是bool，并保证其shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 若输入的Tensor可以广播，则会把低维度通过复制该维度的值的方式扩展到另一个输入中对应的高维度。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入。

    返回：
        Tensor
