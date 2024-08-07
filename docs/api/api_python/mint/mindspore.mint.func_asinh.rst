mindspore.mint.asinh
====================

.. py:function:: mindspore.mint.asinh(input)

    计算输入元素的反双曲正弦。

    .. math::

        out_i = \sinh^{-1}(input_i)

    参数：
        - **input** (Tensor) - 需要计算反双曲正弦函数的输入。

    返回：
        Tensor，数据shape与 `input` 相同。
        当输入类型为bool、int8、uint8、int16、int32、int64时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
