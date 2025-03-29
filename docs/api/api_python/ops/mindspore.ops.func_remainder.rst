mindspore.ops.remainder
=======================

.. py:function:: mindspore.ops.remainder(input, other)

    逐元素计算输入tensor的除法余数。

    支持隐式类型转换、类型提升。

    .. code:: python

        remainder(input, other) == input - input.div(other, rounding_mode="floor") * other

    .. warning::
        - 当输入元素超过2048时，可能会有精度问题。
        - 在Ascend和CPU上的计算结果可能不一致。
        - 如果shape表示为(D1,D2…Dn)，那么D1 \ * D2……\ * DN <= 1000000，n <= 8。

    参数：
        - **input** (Union[Tensor, numbers.Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, numbers.Number, bool]) - 第二个输入。

    返回：
        Tensor
