mindspore.mint.rad2deg
======================

.. py:function:: mindspore.mint.rad2deg(input)

    逐元素地将 `input` 从弧度数制转换为度数制。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入的Tensor。

    返回：
        Tensor，若输入 `input` 为float类型（float16/float32/float64），返回类型与 `input` 相同；
        若输入的 `input` 为uint8、int32、bool类型等，返回的Tensor类型为float32。

    异常：
        - **TypeError** - 如果 `input` 不是一个Tensor。
        - **TypeError** - 如果 `input` 的数据类型为complex64、complex128、uint16、uint32等。
