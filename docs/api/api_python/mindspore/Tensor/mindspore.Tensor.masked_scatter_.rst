mindspore.Tensor.masked_scatter\_
=================================

.. py:method:: mindspore.Tensor.masked_scatter_(mask, source)

    根据 `mask` ，使用 `source` 中的值，更新 `self` 的值，返回一个Tensor。 `mask` 和 `self` 的shape必须相等或者 `mask` 是可广播的。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **mask** (Tensor[bool]) - 一个bool Tensor，其shape可以被广播到 `self` 。
        - **source** (Tensor) - 一个Tensor，其数据类型与 `self` 相同。 `source` 中的元素数量必须大于等于 `mask` 中的True元素的数量。

    返回：
        Tensor，其数据类型和shape与 `self` 相同。