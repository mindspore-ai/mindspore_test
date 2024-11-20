mindspore.Tensor.less_equal
===========================

.. py:method:: Tensor.less_equal(other)

    逐元素计算 :math:`self <= other` 的bool值。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } self_{i}<=other_{i} \\
            & \text{False,   if } self_{i}>other_{i}
            \end{cases}

    .. note::
        - 输入 `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 当 `other` 是一个Scalar时，只能是一个常数。

    参数：
        - **other** (Union[Tensor, Number, bool]) - 数值型，或bool，或数据类型为数值型或bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。

    异常：
       - **TypeError** - 如果 `self` 和 `other` 不是以下之一：Tensor、数值型、bool。
