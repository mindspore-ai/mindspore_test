mindspore.ops.bucketize
==========================

.. py:function:: mindspore.ops.bucketize(input, boundaries, *, right=False)

    返回输入tensor中每个元素所属桶的索引。如果 `right` 为 ``False``，则左边界开放，对于 `input` 中的每个元素 x，返回的索引满足以下规则:

    .. math::

        \begin{cases}
        boundaries[i-1] < x <= boundaries[i], & \text{if right} = False\\
        boundaries[i-1] <= x < boundaries[i], & \text{if right} = True
        \end{cases}

    参数：
        - **input** (Tensor) - 输入tensor。
        - **boundaries** (list) - 存储桶边界值的有序递增列表。

    关键字参数：
        - **right** (bool, 可选) - 如果为 ``False``，则从边界获取输入中每个值的下限索引；如果为 ``True``，则改为获取上限索引。默认 ``False`` 。

    返回：
        Tensor
