mindspore.ops.masked_fill
=========================

.. py:function:: mindspore.ops.masked_fill(input_x, mask, value)

    在掩码为 ``True`` 位置填充指定值。

    支持广播。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **mask** (Tensor[bool]) - 输入掩码。
        - **value** (Union[Number, Tensor]) - 指定值。

    返回：
        Tensor
