mindspore.ops.atanh
====================

.. py:function:: mindspore.ops.atanh(input)

    逐元素计算输入Tensor的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(input_{i})

    参数：
        - **input** (Tensor) - Tensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape与 `input` 相同。当输入数据类型为bool、int8、uint8、int16、int32、int64时，返回值数据类型为float32。否则，返回值数据类型与输入数据类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
