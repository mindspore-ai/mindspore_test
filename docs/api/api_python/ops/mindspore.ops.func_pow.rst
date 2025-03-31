mindspore.ops.pow
==================

.. py:function:: mindspore.ops.pow(input, exponent)

    计算 `input` 中每个元素的 `exponent` 次幂。

    .. note::
        - 支持广播。
        - 支持隐式类型转换、类型提升。

    .. math::

        out_{i} = input_{i} ^{ exponent_{i}}

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入。
        - **exponent** (Union[Tensor, Number]) - 第二个输入。

    返回：
        Tensor
