mindspore.ops.msort
====================

.. py:function:: mindspore.ops.msort(input)

    返回沿第一个维度对输入tensor进行升序排序后的tensor。

    `ops.msort(input)` 等价于 `ops.sort(axis=0)(input)[0]`。更多信息请参考 :func:`mindspore.ops.sort()`。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
