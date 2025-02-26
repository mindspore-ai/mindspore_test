mindspore.Tensor.masked_select
==============================

.. py:method:: mindspore.Tensor.masked_select(mask)

    返回一个一维Tensor，其内容是 `self` 中对应于 `mask` 中True位置的值。 `mask` 的shape与 `self` 的shape不需要一样，但必须符合广播规则。

    参数：
        - **mask** (Tensor[bool]) - 它的shape是 :math:`(x_1, x_2, ..., x_R)`。

    返回：
        一个一维Tensor，类型与 `self` 相同。

    异常：
        - **TypeError** - `mask` 不是Tensor。
        - **TypeError** - `mask` 不是bool类型的Tensor。
