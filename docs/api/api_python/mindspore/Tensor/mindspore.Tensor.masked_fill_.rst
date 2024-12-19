mindspore.Tensor.masked_fill\_
===============================

.. py:method:: mindspore.Tensor.masked_fill_(mask, value)

    将掩码位置为True的位置用指定的值填充 `self` 。 `self` 和 `mask` 的shape需相同或可广播。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **mask** (Tensor[bool]) - 输入的掩码，其数据类型为bool。
        - **value** (Union[Number, Tensor]) - 用来填充的值。

    返回：
        Tensor。

    异常：
        - **TypeError** - `mask` 不是Tensor。
        - **TypeError** - `value` 既不是Number也不是Tensor
        - **RunTimeError** - `self` 或 `value` 的数据类型不支持。
        - **RunTimeError** - `value` 是Tensor却不是0-D Tensor。
        - **RunTimeError** - `self` 和 `mask` 的shape不可广播。
