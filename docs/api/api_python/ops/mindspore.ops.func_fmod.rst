mindspore.ops.fmod
===================

.. py:function:: mindspore.ops.fmod(input, other)

    逐元素计算第一个输入除以第二个输入的余数。

    支持广播、类型提升。

    .. math::
        out = input - n * other

    其中 :math:`n` 是 :math:`input/other` 结果中的整数部分。
    返回值的符号和 `input` 相同，在数值上小于 `other` 。

    参数：
        - **input** (Union[Tensor, Number]) - 被除数。
        - **other** (Union[Tensor, Number]) - 除数。

    返回：
        Tensor
