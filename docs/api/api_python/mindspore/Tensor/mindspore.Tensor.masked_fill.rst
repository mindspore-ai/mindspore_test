mindspore.Tensor.masked_fill
============================

.. py:method:: mindspore.Tensor.masked_fill(mask, value)

    将掩码位置为True的位置填充指定的值。此Tensor和 `mask` 的shape需相同或可广播。

    参数：
        - **mask** (Tensor[bool]) - 输入的掩码，其数据类型为bool。
        - **value** (Union[Number, Tensor]) - 用来填充的值，只支持零维Tensor或者Number。

    返回：
        Tensor，输出与此Tensor的数据类型和shape相同。

    异常：
        - **TypeError** - `mask` 的数据类型不是bool。
        - **TypeError** - `mask` 不是Tensor。
        - **ValueError** - 此Tensor和 `mask` 的shape不可广播。
        - **TypeError** - 此Tensor或 `value` 的数据类型不是bool、int8、int32、int64、float16、float32或bfloat16。
        - **TypeError** - `value` 的数据类型与此Tensor不同。
        - **TypeError** - `value` 既不是Number也不是Tensor。
