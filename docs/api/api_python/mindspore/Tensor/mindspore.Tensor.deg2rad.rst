mindspore.Tensor.deg2rad
=========================

.. py:method:: mindspore.Tensor.deg2rad()

    逐元素地将 `self` 从度数制转换为弧度制。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        `self` 支持数据类型如下：

        - Ascend：float16、float32、float64、uint8、int8、int16、int32、int64、bool。
        - CPU/GPU：float16、float32、float64。

    返回：
        Tensor，若输入 `self` 为float类型（float16/float32/float64），返回类型与 `self` 相同；
        若输入的 `self` 为uint8、int32、bool类型等，返回的Tensor类型为float32。

    异常：
        - **TypeError** - 如果 `self` 的数据类型为complex64、complex128、uint16、uint32等。
