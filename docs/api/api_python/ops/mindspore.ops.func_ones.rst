mindspore.ops.ones
===================

.. py:function:: mindspore.ops.ones(shape, dtype=None)

    创建一个值全为1的tensor。

    .. warning::
        参数 `shape` 在后续版本中将不再支持Tensor类型的输入。

    参数：
        - **shape** (Union[tuple[int], list[int], int, Tensor]) - 指定的shape。
        - **dtype** (:class:`mindspore.dtype`) - 指定数据类型，默认 ``None`` 。

    返回：
        Tensor