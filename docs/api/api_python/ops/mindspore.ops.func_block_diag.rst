mindspore.ops.block_diag
=========================

.. py:function:: mindspore.ops.block_diag(*inputs)

    基于输入tensor创建块对角矩阵。

    参数：
        - **inputs** (Tensor) - 一个或多个tensor，tensor的维度应该为0、1或2。

    返回：
        二维tensor。所有输入tensor按顺序排列，使其左上角和右下角对角线相邻，其他所有元素都置零。
