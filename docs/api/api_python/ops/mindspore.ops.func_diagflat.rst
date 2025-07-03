mindspore.ops.diagflat
======================

.. py:function:: mindspore.ops.diagflat(input, offset=0)

    如果input是一维tensor，则返回input作为对角线的二维tensor，如果input是大于等于二维的tensor，则将input展开后作为对角线的二维tensor。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **offset** (int, 可选) - 对角线偏移量。默认 ``0`` 。

          - 当 `offset` 是正整数时，对角线向上方偏移。
          - 当 `offset` 是负整数时，对角线向下方偏移。

    返回：
        Tensor
