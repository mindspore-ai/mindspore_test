mindspore.ops.cholesky_solve
============================

.. py:function:: mindspore.ops.cholesky_solve(input, input2, upper=False)

    根据Cholesky分解因子 `input2` 计算一组具有正定矩阵的线性方程组的解。

    如果 `upper` 为 ``True``， `input2` 是上三角矩阵，输出的结果：

    .. math::
        output = (input2^{T} * input2)^{{-1}}input

    如果 `upper` 为 ``False``， `input2` 是下三角矩阵，输出的结果：

    .. math::
        output = (input2 * input2^{T})^{{-1}}input

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - shape为 :math:`(*, N, M)` 的输入tensor。
        - **input2** (Tensor) - shape为 :math:`(*, N, N)` 的tensor，Cholesky因子。
        - **upper** (bool, 可选) - 是否视为上三角矩阵。默认 ``False`` 。

    返回：
        Tensor