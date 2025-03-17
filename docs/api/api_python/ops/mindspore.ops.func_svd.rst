mindspore.ops.svd
==================

.. py:function:: mindspore.ops.svd(input, full_matrices=False, compute_uv=True)

    计算单个或多个矩阵的奇异值分解。

    设矩阵 :math:`A` ，svd返回奇异值 :math:`S` 、左奇异向量 :math:`U` 和右奇异向量 :math:`V` 。满足以下公式：

    .. math::
        A=U*diag(S)*V^{T}

    参数：
        - **input** (Tensor) - 待分解的矩阵。shape为 :math:`(*, M, N)` 。
        - **full_matrices** (bool, 可选) - 如果为 ``True`` ，则计算完整的 :math:`U` 和 :math:`V` 。否则仅计算前P个奇异向量，P为M和N中的较小值，M和N分别是输入矩阵的行和列。默认值： ``False`` 。
        - **compute_uv** (bool, 可选) - 如果此参数的值为 ``True`` ，则计算 :math:`U` 和 :math:`V` ，否则只计算 :math:`S` 。默认值： ``True`` 。

    返回：
        如果 `compute_uv` 为 ``True`` ，将返回包含 `s` 、 `u` 和 `v` 的Tensor元组。否则，将仅返回单个Tensor `s` 。

        - `s` 是奇异值Tensor。shape为 :math:`(*, P)` 。
        - `u` 是左奇异向量Tensor。如果 `compute_uv` 为 ``False`` ，该值不会返回。shape为 :math:`(*, M, P)` 。如果 `full_matrices` 为 ``True`` ，则shape为 :math:`(*, M, M)` 。
        - `v` 是右奇异向量Tensor。如果 `compute_uv` 为 ``False`` ，该值不会返回。shape为 :math:`(*, N, P)` 。如果 `full_matrices` 为 ``True`` ，则shape为 :math:`(*, N, N)` 。

    异常：
        - **TypeError** - `full_matrices` 或 `compute_uv` 不是bool类型。
        - **TypeError** - 输入的rank小于2。
