mindspore.ops.bmm
=================

.. py:function:: mindspore.ops.bmm(input_x, mat2)

    对输入的两个tensor执行批量矩阵乘积。

    .. math::
        \text{output}[..., :, :] = \text{matrix}(input_x[..., :, :]) * \text{matrix}(mat2[..., :, :])

    `input_x` 的维度不能小于 `3` ， `mat2` 的维度不能小于 `2` 。

    参数：
        - **input_x** (Tensor) - 输入的第一个Tensor。
        - **mat2** (Tensor) - 输入的第二个Tensor。

    返回：
        Tensor
