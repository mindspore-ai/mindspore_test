mindspore.ops.minimum
=====================

.. py:function:: mindspore.ops.minimum(input, other)

    逐元素计算两个输入tensor的最小值。

    .. math::
        output_i = \min(input_i, other_i)

    .. note::
        - 输入 `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 当输入是两个Tensor时，它们的数据类型不能同时是bool。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播操作。
        - 当一个元素与NaN比较时，将返回NaN。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入。

    返回：
        Tensor
