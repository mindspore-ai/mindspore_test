mindspore.ops.triu_indices
==========================

.. py:function:: mindspore.ops.triu_indices(row, col, offset=0, *, dtype=mstype.int64)

    返回一个二维的tensor，包含 `row` * `col` 矩阵的上三角元素的索引。第一行是所有索引的行坐标，第二行是所有索引的列坐标。
    坐标先按行排序，后按列排序。

    .. note::
        在CUDA上运行的时候， `row` * `col` 必须小于2^59以防止计算时溢出。

    参数：
        - **row** (int) - 二维tensor的行数。
        - **col** (int) - 二维tensor的列数。
        - **offset** (int, 可选) - 对角线偏移量。默认 ``0`` 。

          - 当 `offset` 是正整数时，对角线向上方偏移。
          - 当 `offset` 是负整数时，对角线向下方偏移。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型，支持 `mstype.int32` 和 `mstype.int64` ，默认 ``mstype.int64`` 。

    返回：
        Tensor
