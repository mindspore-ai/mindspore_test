mindspore.ops.subtract
=======================

.. py:function:: mindspore.ops.subtract(input, other, *, alpha=1)

    从 `input` 中减去经 `alpha` 缩放的 `other`。

    支持隐式类型转换、类型提升。

    .. math::
        output[i] = input[i] - alpha * other[i]

    参数：
        - **input** (Union[Tensor, number.Number]) - 第一个输入。
        - **other** (Union[Tensor, number.Number]) - 第二个输入。

    关键字参数：
        - **alpha** (number) - :math:`other` 的乘数。默认 ``1`` 。

    返回：
        Tensor
