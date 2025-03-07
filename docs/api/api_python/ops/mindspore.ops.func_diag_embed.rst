mindspore.ops.diag_embed
=========================

.. py:function:: mindspore.ops.diag_embed(input, offset=0, dim1=-2, dim2=-1)

    创建一个tensor，其特定二维平面（由 `dim1` 和 `dim2` 指定）的对角线由输入tensor填充，其余位置填充为 ``0`` 。不指定维度时，默认填充返回tensor的最后两个维度所形成的二维平面的对角线。

    参数：
        - **input** (Tensor) - 对角线填充值。
        - **offset** (int，可选) - 对角线偏移量。默认 ``0`` 。

          - 当 `offset` 是正整数时，对角线向上方偏移。
          - 当 `offset` 是负整数时，对角线向下方偏移。
        - **dim1** (int，可选) - 填充对角线的第一个维度。默认 ``-2`` 。
        - **dim2** (int，可选) - 填充对角线的第二个维度。默认 ``-1`` 。

    返回：
        一个数据类型与 `input` 一致，但输出shape维度比 `input` 高一维的tensor。

    异常：
        - **ValueError** - `input` 的维度不是1D-6D。