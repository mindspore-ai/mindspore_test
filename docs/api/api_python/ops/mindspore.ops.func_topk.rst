mindspore.ops.topk
===================

.. py:function:: mindspore.ops.topk(input, k, dim=None, largest=True, sorted=True)

    按指定维度返回前 `k` 个最大或最小元素及其对应索引。

    .. warning::
        - 如果 `sorted` 设置为False，它将使用aicpu运算符，性能可能会降低。此外，由于在不同平台上存在内存排布以及遍历方式不同等问题，`sorted` 设置为False时计算结果的显示顺序可能会出现不一致的情况。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **k** (int) - 返回的元素数量。
        - **dim** (int, 可选) - 指定维度。如果为 ``None``，则会按最后一个维度排序。默认 ``None`` 。
        - **largest** (bool, 可选) - 如果为 ``True`` ，返回最大元素。如果为 ``False`` ，返回最小元素。默认 ``True`` 。
        - **sorted** (bool, 可选) - 如果为 ``True`` ，则返回的元素按降序排序。如果为 ``False`` ，则不对获取的元素进行排序。默认 ``True`` 。

    返回：
        两个tensor组成的tuple(values, indices)
