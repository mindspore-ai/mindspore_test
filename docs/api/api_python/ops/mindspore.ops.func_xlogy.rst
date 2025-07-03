mindspore.ops.xlogy
====================

.. py:function:: mindspore.ops.xlogy(input, other)

    逐元素对 `other` 取对数，再与 `input` 相乘。

    .. math::
        out_i = input_{i} * ln{other_{i}}

    .. note::
        - 支持广播，支持隐式类型转换、类型提升。

    .. warning::
        Ascend平台， `input` 和 `other` 必须为float16或float32。

    参数：
        - **input** (Union[Tensor, numbers.Number, bool]) - 第一个输入tensor。
        - **other** (Union[Tensor, numbers.Number, bool]) - 第二个输入tensor。

    返回：
        Tensor
