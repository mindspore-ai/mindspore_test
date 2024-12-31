mindspore.Tensor.floor
======================

.. py:method:: mindspore.Tensor.floor()

    逐元素向下取整函数。

    .. math::
        out_i = \lfloor input_i \rfloor

    返回：
        Tensor，shape与 `self` 相同。

    异常：
        - **TypeError** - `self` 的数据类型不支持。 支持的数据类型：

          - Ascend：float16、float32、float64、bfloat16、int8、int16、int32、int64、uint8、uint16、uint32、uint64。
          - GPU/CPU：float16、float32、float64。
