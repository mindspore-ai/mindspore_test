mindspore.ops.float_power
==========================

.. py:function:: mindspore.ops.float_power(input, exponent)

    逐元素计算以第一个输入为底，第二个输入为指数。如果输入为实数类型，则转换为mindspore.float64计算。

    .. note::
        目前不支持复数运算。

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入。
        - **exponent** (Union[Tensor, Number]) - 第二个输入。如果 `input` 是Number，则该参数必须是Tensor类型。

    返回：
        类型为mindspore.float64的tensor。

