mindspore.Tensor.maximum
========================

.. py:method:: mindspore.Tensor.maximum(other)

    逐元素计算两个输入Tensor中的最大值。

    .. math::
        output_i = \max(tensor_i, other_i)

    .. note::
        - 输入 `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入 `other` 可以是一个tensor或一个scalar。
        - 当输入 `other` 是Tensor时， `self` 和 `other` 的数据类型不能同时是bool，并保证其shape可以广播。
        - 当输入是一个Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 如果一个元素和NaN比较，则返回NaN。

    .. warning::
        如果所有输入都为标量int类型，在Graph模式下，输出为int32类型的Tensor，在PyNative模式下，输出为int64类型的Tensor。

    参数：
        - **other** (Union[Tensor, Number, bool]) - 输入可以是Number或bool，也可以是数据类型为Number或bool的Tensor。

    返回：
        Tensor的shape与广播后的shape相同，数据类型为两个输入中精度较高或数字较多的类型。

    异常：
        - **TypeError** - `other` 不是以下之一：Tensor、Number、bool。
        - **ValueError** - `self` 和 `other` 的shape不相同。
