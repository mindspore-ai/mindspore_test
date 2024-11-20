mindspore.Tensor.logical_or
===========================

.. py:method:: Tensor.logical_or(other)

    逐元素计算两个Tensor的逻辑或运算。

    .. math::
        out_{i} = self_{i} \\vee other_{i}

    .. note::
        - `self` 和 `other` 的输入遵循隐式类型转换规则，使数据类型一致。
        - `other` 是一个bool时，bool对象只能是一个常量。

    输入：
        - **other** (Union[Tensor, bool]) - bool或者数据类型可被隐式转换为bool的Tensor。

    输出：
        Tensor，其shape与广播后的shape相同，数据类型为bool。
