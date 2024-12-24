mindspore.Tensor.acos
=====================

.. py:method:: mindspore.Tensor.acos()

    逐元素计算 `self` 的反余弦。

    .. math::
        out_i = \cos^{-1}(input_i)

    返回：
        Tensor，shape与 `self` 相同。当 `self` 数据类型为bool、int8、uint8、int16、int32、int64时，返回值数据类型为float32。否则，返回值数据类型与 `self` 数据类型相同。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
