mindspore.ops.logical_or
==============================

.. py:function:: mindspore.ops.logical_or(input, other)

    逐元素计算两个tensor的逻辑或运算。

    .. math::
        out_{i} = input_{i} \vee other_{i}

    .. note::
        - 支持广播。
        - 支持隐式类型转换。

    参数：
        - **input** (Union[Tensor, bool]) - 第一个输入。
        - **other** (Union[Tensor, bool]) - 第二个输入。

    返回：
        Tensor
