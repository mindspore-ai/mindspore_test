mindspore.ops.full
==================

.. py:function:: mindspore.ops.full(size, fill_value, *, dtype=None)

    创建tensor，用指定值填充。

    .. note::
        `fill_value` 不支持复数类型。

    参数：
        - **size** (Union(tuple[int], list[int])) - 指定的shape。
        - **fill_value** (number.Number) - 指定值。

    关键字参数：
        - **dtype** (mindspore.dtype) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
