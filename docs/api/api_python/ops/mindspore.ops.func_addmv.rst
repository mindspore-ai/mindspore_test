mindspore.ops.addmv
======================

.. py:function:: mindspore.ops.addmv(input, mat, vec, *, beta=1, alpha=1)

    将矩阵 `mat` 和 向量 `vec` 相乘，结果与输入 `input` 相加。

    .. note::
        - 如果 `mat` 是一个大小为 :math:`(N, M)` 的tensor，且 `vec` 是一个大小为 :math:`M` 的一维tensor，那么 `input` 必须是可广播的，并且是一个大小为 :math:`N` 的一维tensor。
        - 若 `beta` 为0，那么 `input` 将被忽略。

    .. math::
        output = β input + α (mat @ vec)

    参数：
        - **input** (Tensor) - 输入tensor。
        - **mat** (Tensor) - 将被乘的矩阵tensor。
        - **vec** (Tensor) - 将被乘的向量tensor。

    关键字参数：
        - **beta** (scalar[int, float, bool], 可选) - `input` 的尺度因子，默认 ``1`` 。
        - **alpha** (scalar[int, float, bool], 可选) - （ `mat` @ `vec` ）的尺度因子，默认 ``1`` 。

    返回：
        Tensor
