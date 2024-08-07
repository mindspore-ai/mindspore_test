mindspore.mint.acosh
====================

.. py:function:: mindspore.mint.acosh(input)

    逐元素计算输入Tensor的反双曲余弦。

    .. math::
        out_i = \cosh^{-1}(input_i)

    .. warning::
        给定一个输入Tensor `input` ，该函数计算每个元素的反双曲余弦。输入范围为[1, inf]。

    参数：
        - **input** (Tensor) - 需要计算反双曲余弦函数的输入Tensor。

    返回：
        Tensor，数据shape与 `input` 相同。
        当输入类型为bool、int8、uint8、int16、int32、int64时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
