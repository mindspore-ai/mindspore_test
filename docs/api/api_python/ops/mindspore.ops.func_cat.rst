mindspore.ops.cat
==================

.. py:function:: mindspore.ops.cat(tensors, axis=0)

    在指定轴上拼接输入tensor。

    参数：
        - **tensors** (Union[tuple[Tensor], list[Tensor]]) - 输入tensors。除了指定的拼接轴 `axis` 之外，其他轴的shape都应相等。
        - **axis** (int) - 指定轴。默认 ``0`` 。

    返回：
        Tensor
