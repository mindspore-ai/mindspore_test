mindspore.ops.copysign
=======================

.. py:function:: mindspore.ops.copysign(x, other)

    创建一个float tensor,由 `x` 的绝对值和 `other` 的符号组成。

    支持广播。

    参数：
        - **x** (Union[Tensor]) - 输入tensor
        - **other** (Union[int, float, Tensor]) - 决定返回值符号的tensor。

    返回：
        Tensor