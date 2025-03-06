mindspore.ops.floor_divide
==========================

.. py:function:: mindspore.ops.floor_divide(input, other)

    逐元素计算 `input` 除以 `other` ，并向下取整。

    如果 `input` 和 `other` 数据类型不同，遵循隐式类型转换规则。
    输入必须是两个tensor或一个tensor和一个scalar。
    当输入是两个tensor时，其shape须可以进行广播，并且数据类型不能同时为bool。

    .. math::
        out_{i} = \text{floor}( \frac{input_i}{other_i})

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入tensor。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入tensor。

    返回：
        Tensor
