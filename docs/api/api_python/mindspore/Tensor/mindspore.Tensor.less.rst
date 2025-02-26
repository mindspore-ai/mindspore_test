mindspore.Tensor.less
=====================

.. py:method:: mindspore.Tensor.less(other)

    逐元素计算 :math:`self < other` ，返回为bool。

    `self` 和 `other` 的输入遵循隐式类型转换规则，使数据类型一致。
    当 `other` 是一个Scalar时，只能是一个常量。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } self_{i}<other_{i} \\
            & \text{False,   if } self_{i}>=other_{i}
            \end{cases}

    参数：
        - **other** (Union[Tensor, Number, bool]) - 待比较的值。支持数值型、bool或Tensor[Number/bool]。

    返回：
        Tensor，输出shape与广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - 如果 `self` 和 `other` 不是以下之一：Tensor、数值型、bool。
