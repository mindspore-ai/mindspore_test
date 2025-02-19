mindspore.Tensor.eq
===================

.. py:method:: mindspore.Tensor.eq(other)

    逐元素比较两个输入Tensor是否相等。

    第二个参数可以是一个shape，也可以广播成第一个参数的Number或Tensor，反之亦然。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } self_{i} = other_{i} \\
            & \text{False,   if } self_{i} \ne other_{i}
            \end{cases}

    .. note::
        - `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 另一个输入必须是一个Tensor或Scalar。
        - 两个输入的shape支持广播。

    参数：
        - **other** (Union[Tensor, Number]) - 另一个输入可以是数值型，也可以是数据类型为数值型的Tensor。

    返回：
        Tensor，输出的shape与输入广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - `self` 和 `other` 都不是Tensor。
