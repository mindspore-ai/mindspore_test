mindspore.ops.tan
===================

.. py:function:: mindspore.ops.tan(input)

    逐元素计算输入元素的正切值。

    .. math::
        out_i = \tan(input_i)

    参数：
        - **input** (Tensor) - Tan的输入，任意维度的Tensor。

    返回：
        Tensor，shape与 `input` 相同。当输入数据类型为bool、int8、uint8、int16、int32、int64时，返回值数据类型为float32。否则，返回值数据类型与输入数据类型相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
