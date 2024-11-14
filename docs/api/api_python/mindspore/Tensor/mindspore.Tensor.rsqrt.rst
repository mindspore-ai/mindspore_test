mindspore.Tensor.rsqrt
=======================

.. py:method:: mindspore.Tensor.rsqrt()

    逐元素计算输入Tensor元素的平方根倒数。

    .. math::
        out_{i} =  \frac{1}{\sqrt{input_{i}}}

    返回：
        Tensor，具有与 `self` 相同的shape。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
