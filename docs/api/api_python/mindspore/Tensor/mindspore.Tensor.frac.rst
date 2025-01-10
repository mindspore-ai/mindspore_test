mindspore.Tensor.frac
=====================

.. py:method:: mindspore.Tensor.frac()

    计算 `self` 中每个元素的小数部分。

    .. math::
        out_i = self_i - \lfloor |self_i| \rfloor * sgn(self_i)

    返回：
        Tensor，其类型和shape与 `self` 相同。

    异常：
        - **TypeError** - `self` 不是Tensor。
