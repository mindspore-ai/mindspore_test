mindspore.Tensor.cosh
=====================

.. py:method:: mindspore.Tensor.cosh()

    逐元素计算 `self` 的双曲余弦值。

    .. math::
        out_i = \cosh(self_i)

    返回：
        Tensor，数据类型和shape与 `self` 相同。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
        - **TypeError** - 

          - CPU/GPU: 如果 `self` 的数据类型不是float16、float32、float64、complex64或complex128。
          - Ascend: 如果 `self` 的数据类型不是bool、int8、uint8、int16、int32、int64、float16、float32、float64、complex64、complex128或bfloat16。
