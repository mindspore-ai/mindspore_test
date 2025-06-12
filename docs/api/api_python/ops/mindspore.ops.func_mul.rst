mindspore.ops.mul
=================

.. py:function:: mindspore.ops.mul(input, other)

    两个Tensor逐元素相乘。

    .. math::

        out_{i} = input_{i} * other_{i}

    .. note::
        - 当两个输入具有不同的shape时，它们的shape必须能够广播为一个共同的shape。
        - 两个输入不能同时为bool类型。[True, Tensor(True), Tensor(np.array([True]))]等都属于bool类型。
        - 支持隐式类型转换、类型提升。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入。

    返回：
        Tensor
