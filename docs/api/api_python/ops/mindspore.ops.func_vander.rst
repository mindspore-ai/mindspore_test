mindspore.ops.vander
====================

.. py:function:: mindspore.ops.vander(x, N=None)

    生成一个范德蒙矩阵。
    返回矩阵的各列是入参各元素的幂。第 i 个输出列是输入向量元素的幂，其幂为 :math:`N - i - 1`。

    参数：
        - **x** (Tensor) - 一维输入tensor。
        - **N** (int，可选) - 返回矩阵的列数。默认 ``None`` ，为 :math:`len(x)`。

    返回：
        Tensor
