mindspore.Tensor.greater_equal
==============================

.. py:method:: mindspore.Tensor.greater_equal(other)

    逐元素计算 :math:`self >= other` 的bool值。

    参数：
        - **other** (Union[Tensor, Number]) - `other` 应该是一个Number或数据类型为number或bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。
