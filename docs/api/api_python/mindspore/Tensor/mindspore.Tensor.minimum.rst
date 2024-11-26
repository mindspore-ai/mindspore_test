mindspore.Tensor.minimum
========================

.. py:method:: mindspore.Tensor.minimum(other)

    逐元素计算两个输入Tensor中的最小值。

    .. math::
        output_i = \min(tensor_i, other_i)

    .. note::
        - 输入 `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入 `other` 可以是一个tensor或一个scalar。
        - 当输入 `other` 是Tensor时， `self` 和 `other` 的数据类型不能同时是bool，并保证其shape可以广播。
        - 当 `other` 是一个Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 如果一个元素和NaN比较，则返回NaN。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - `other` 可以是number.Number或bool，也可以是数据类型为number.Number或bool的Tensor。

    返回：
        Tensor，其shape与广播后的shape相同，其数据类型为 `self` 和 `other` 中精度较高的类型。

    异常：
        - **TypeError** - `other` 不是以下之一：Tensor、Number、bool。
        - **ValueError** - `self` 和 `other` 广播后的shape不相同。
