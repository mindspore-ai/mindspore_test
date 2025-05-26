mindspore.Tensor.bitwise_or
===========================

.. py:method:: mindspore.Tensor.bitwise_or(other) -> Tensor

    逐元素执行两个Tensor的或运算。

    .. note::
        参数 `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **other** (Tensor, Number.number) - 输入Tensor或常量，shape与 `self` 相同，或能与 `self` 的shape广播。

    返回：
       Tensor，与广播后的输入shape相同，和 `self` 数据类型相同。
