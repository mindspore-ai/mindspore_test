mindspore.ops.gcd
=================

.. py:function:: mindspore.ops.gcd(input, other)

    逐元素计算两个输入tensor的最大公约数。

    支持广播、类型提升。数据类型支持：int16（使用Ascend后端时支持，graph mode只在jit level为O0时支持）、int32、int64之一。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。

    返回：
        Tensor
