mindspore.mint.triangular_solve
===============================

.. py:function:: mindspore.mint.triangular_solve(b, A, upper=True, transpose=False, unitriangular=False)

    求解正上三角形或下三角形可逆矩阵 `A` 和包含多个元素的右侧边 `b` 的方程组的解。

    用符号表示，它求解方程 :math:`A X = b`，并假设矩阵 `A` 是一个方阵，且为上三角矩阵（如果 ``upper = False``，则为下三角矩阵），并且其对角线上没有零元素。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **b** (Tensor) - shape是 :math:`(*, M, K)` 的Tensor，其中*表示任意数量的维度。
        - **A** (Tensor) - shape是 :math:`(*, M, M)` 的Tensor，其中*表示任意数量的维度。
        - **upper** (bool，可选) - 矩阵 `A` 是为上三角矩阵或下三角矩阵。默认值：``True``。
        - **transpose** (bool，可选) - 求解方程 :math:`op(A) X = b`，其中如果此标志为 True，则 :math:`op(A) = A^T`；如果为 False，则 :math:`op(A) = A`。默认值：``False``。
        - **unitriangular** (bool，可选) - 矩阵 `A` 是否为单位三角矩阵。如果为 True，则假设矩阵 `A` 的对角线元素为 1，并且不会从矩阵 `A` 中引用这些对角线元素。默认值：``False``。

    返回：
        包含 `X` 和 `A` 的tuple。

    异常：
        - **TypeError** - 如果参数 `b` 不是Tensor。
        - **TypeError** - 如果参数 `A` 不是Tensor。
        - **TypeError** - 如果 `upper` 不是bool。
        - **TypeError** - 如果 `transpose` 不是bool。
        - **TypeError** - 如果 `unitriangular` 不是bool。
        - **ValueError** - 如果 `b` 或者 `A` 的维度不在 :math:`[2, 6]` 范围内。
        - **ValueError** - 如果 `b` 和 `A` 的shape不匹配。