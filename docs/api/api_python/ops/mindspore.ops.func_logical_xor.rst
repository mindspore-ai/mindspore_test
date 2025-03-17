mindspore.ops.logical_xor
=========================

.. py:function:: mindspore.ops.logical_xor(input, other)

    逐元素计算两个tensor的逻辑异或运算。

    .. math::
        out_{i} = input_{i} \oplus other_{i}

    .. note::
        - 支持广播。
        - 支持隐式类型转换。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。

    返回：
        Tensor
