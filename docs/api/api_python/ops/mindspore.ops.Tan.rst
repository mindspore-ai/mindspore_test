mindspore.ops.Tan
===================

.. py:class:: mindspore.ops.Tan

    逐元素计算输入元素的正切值。

    更多参考详见 :func:`mindspore.ops.tan`。

    输入：
        - **input** (Tensor) - 任意维度的输入Tensor。

    输出：
        Tensor，数据shape与 `input` 相同。
        当输入类型为bool、int8、uint8、int16、int32、int64时，返回值类型为float32。
        否则，返回值类型与输入类型相同。
