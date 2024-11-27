mindspore.Tensor.sqrt
=====================

.. py:method:: mindspore.Tensor.sqrt()

    逐元素计算 `self` 的平方根。

    .. note::
        当 `self` 的值中存在一些负数，则负数对应位置上的返回结果为NaN。

    .. math::
        out_{i} =  \sqrt{self_{i}}

    返回：
        Tensor，shape和数据类型与输入 `self` 相同。
