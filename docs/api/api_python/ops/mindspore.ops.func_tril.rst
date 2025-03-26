mindspore.ops.tril
===================

.. py:function:: mindspore.ops.tril(input, diagonal=0)

    将指定对角线上方的元素设置为0。

    参数：
        - **input** (Tensor) - 输入tensor，其秩至少为2。
        - **diagonal** (int，可选) - 二维tensor的指定对角线。默认 ``0`` ，表示主对角线。

    返回：
        Tensor