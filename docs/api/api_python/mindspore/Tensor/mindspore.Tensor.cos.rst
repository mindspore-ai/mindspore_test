mindspore.Tensor.cos
====================

.. py:method:: Tensor.cos()

    逐元素计算输入的余弦。

    .. math::
        out_i = \cos(self_i)

    .. warning::
        如果使用float64，可能会存在精度丢失的问题。

    返回：
        Tensor，shape与 `self` 相同。
        当输入类型为bool、int8、uint8、int16、int32、int64时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
        - **TypeError** -

          - CPU/GPU: 如果 `self` 的数据类型不是float16、float32、float64、complex64或complex128。
          - Ascend: 如果 `self` 的数据类型不是bool、int8、uint8、int16、int32、int64、float16、float32、float64、complex64或complex128。