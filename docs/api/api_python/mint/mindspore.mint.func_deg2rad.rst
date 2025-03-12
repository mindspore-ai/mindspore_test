mindspore.mint.deg2rad
======================

.. py:function:: mindspore.mint.deg2rad(input)

    逐元素地将 `input` 从度数制转换为弧度制。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入的Tensor。

    返回：
        Tensor，若输入 `input` 为float类型（float16/float32/float64），返回类型与 `input` 相同；
        若输入的 `input` 为uint8、int8、int16、bool类型等，返回的Tensor类型为float32。

    异常：
        - **TypeError** - 如果 `input` 不是一个Tensor。
        - **TypeError** - 如果 `input` 的数据类型为complex64、complex128、uint16、uint32等。
