mindspore.Tensor.le
===================

.. py:method:: mindspore.Tensor.le(other)

    逐元素计算 :math:`self <= other` 的bool值。

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } self_{i}<=other_{i} \\
            & \text{False,   if } self_{i}>other_{i}
            \end{cases}

    .. note::
        - 输入 `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入 `other` 是Tensor或Scalar, 当 `other` 是一个Scalar时，Scalar只能是一个常数。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - 输入 `other` 应该是一个number.Number或bool值，或数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。
