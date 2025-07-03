mindspore.ops.full_like
=======================

.. py:function:: mindspore.ops.full_like(input, fill_value, *, dtype=None)

    返回一个用指定值填充的tensor，shape与 `input` 相同。

    .. note::
        `fill_value` 不支持复数类型。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **fill_value** (Number) - 指定值。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor