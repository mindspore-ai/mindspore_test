mindspore.mint.addmv
====================

.. py:function:: mindspore.mint.addmv(input, mat, vec, *, beta=1, alpha=1)

    `mat` 和 `vec` 矩阵向量相乘，且将输入向量 `input` 加到最终结果中。

    如果 `mat` 是一个大小为 :math:`(N, M)` Tensor， `vec` 是一个大小为 :math:`M` 的一维Tensor，那么 `input` 必须\
    可广播到一个大小为 :math:`N` 的一维Tensor。这种情况下 `output` 是一个大小为 :math:`N` 的一维Tensor。

    .. math::
        output = \beta input + \alpha (mat @ vec)

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 被加的向量。
        - **mat** (Tensor) - 第一个需要相乘的Tensor。
        - **vec** (Tensor) - 第二个需要相乘的Tensor。

    关键字参数：
        - **beta** (Union[float, int], 可选) - 输入的系数。默认值： ``1`` 。
        - **alpha** (Union[float, int]，可选) - :math:`mat @ vec` 的系数。默认值： ``1`` 。

    返回：
        Tensor，shape大小为 :math:`(N,)` ，其数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 、 `mat` 或 `vec` 不是Tensor。
        - **TypeError** - `mat` 和 `vec` 数据类型不一致。
        - **ValueError** - `mat` 不是二维Tensor。
        - **ValueError** - `vec` 不是一维Tensor。
