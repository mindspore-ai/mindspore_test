mindspore.Tensor.not_equal
===========================

.. py:method:: mindspore.Tensor.not_equal(other)

    计算两个Tensor是否不相等。

    .. math::
        out_{i} =\begin{cases}
        & \text{True,    if } tensor_{i} \ne other_{i} \\
        & \text{False,   if } tensor_{i} = other_{i}
        \end{cases}

    .. note::
        - 输入 `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入 `other` 可以是一个tensor或一个scalar。
        - 当 `other` 是Tensor时，`self` 和 `other` 的shape可以广播。
        - 当 `other` 是一个Scalar时，Scalar只能是一个常数。
        - 支持广播。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - `other` 可以是number.Number或bool，也可以是数据类型为number.Number或bool的Tensor。

    返回：
        Tensor的shape与广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - `other` 不是以下之一：Tensor、数值型、bool。

