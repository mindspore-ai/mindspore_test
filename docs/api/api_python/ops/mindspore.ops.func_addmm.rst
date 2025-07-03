mindspore.ops.addmm
====================

.. py:function:: mindspore.ops.addmm(input, mat1, mat2, *, beta=1, alpha=1)

    对 `mat1` 和 `mat2` 矩阵乘，再将结果与 `input` 相加。

    .. note::
        - 若 `beta` 为0，那么 `input` 将会被忽略。

    .. math::
        output = \beta input + \alpha (mat1 @ mat2)

    参数：
        - **input** (Tensor) - 输入tensor。
        - **mat1** (Tensor) - 第一个矩阵。
        - **mat2** (Tensor) - 第二个矩阵。

    关键字参数：
        - **beta** (Union[int, float]，可选) - `input` 的尺度因子。默认 ``1`` 。
        - **alpha** (Union[int, float]，可选) - （ `mat1` @ `mat2` ）的尺度因子。默认 ``1`` 。

    返回：
        Tensor
