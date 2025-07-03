mindspore.ops.randint
======================

.. py:function:: mindspore.ops.randint(low, high, size, seed=None, *, dtype=None)

    返回一个tensor，其元素为 [ `low` , `high` ) 区间的随机整数。

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **low** (int) - 随机区间的起始值。
        - **high** (int) - 随机区间的末尾值。
        - **size** (tuple) - 输出tensor的shape。
        - **seed** (int，可选) - 随机种子，必须大于或等于0。默认 ``None`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
