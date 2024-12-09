mindspore.Tensor.greater_equal
==============================

.. py:method:: mindspore.Tensor.greater_equal(other)

    逐元素计算 :math:`self >= other` 的bool值。

    参数：
        - **other** (Union[Tensor, Number]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个Number或数据类型为number或bool_的Tensor。当第一个输入是Scalar时，第二个输入必须是数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。
