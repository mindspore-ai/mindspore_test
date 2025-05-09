mindspore.ops.add
=================

.. py:function:: mindspore.ops.add(input, other)

    逐元素计算两个输入tensor的和。

    .. math::

        out_{i} = input_{i} + other_{i}

    .. note::
        - 两个输入不能同时为bool类型。[True, Tensor(True), Tensor(np.array([True]))]等都为bool类型。
        - 支持广播，支持隐式类型转换、类型提升。
        - 当输入为tensor时，维度应大于等于1。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入tensor。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入tensor。
    返回：
        Tensor
