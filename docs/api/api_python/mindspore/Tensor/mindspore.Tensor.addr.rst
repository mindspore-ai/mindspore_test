mindspore.Tensor.addr
=====================

.. py:method:: mindspore.Tensor.addr(vec1, vec2, *, beta=1, alpha=1)

    计算 `vec1` 和 `vec2` 的外积，并将其添加到 `self` 中。

    如果 `vec1` 是一个大小为 :math:`N` 的向量， `vec2` 是一个大小为 :math:`M` 的向量，那么 `self` 必须可以和大小为 :math:`(N, M)` 的矩阵广播。

    可选值 `bata` 和 `alpha` 分别是 `vec1` 和 `vec2` 外积以及附加矩阵 `self` 的扩展因子。如果 `beta` 为0，那么 `self` 将不参与计算。

    .. math::
        output = \beta self + \alpha (vec1 \otimes vec2)

    参数：
        - **vec1** (Tensor) - 第一个需要相乘的Tensor，shape大小为 :math:`(N,)` 。
        - **vec2** (Tensor) - 第二个需要相乘的Tensor，shape大小为 :math:`(M,)` 。

    关键字参数：
        - **beta** (scalar[int, float, bool], 可选) - `self` 的乘数。 `beta` 必须是int或float或bool类型，默认值： ``1`` 。
        - **alpha** (scalar[int, float, bool], 可选) - :math:`vec1 \otimes vec2` 的乘数。 `alpha` 必须是int或float或bool类型，默认值： ``1`` 。

    返回：
        Tensor，shape大小为 :math:`(N, M)` ，数据类型与 `self` 相同。

    异常：
        - **TypeError** - `self` 、 `vec1` 、 `vec2` 不是Tensor。
        - **TypeError** - `vec1` 、 `vec2` 的数据类型不一致。
        - **ValueError** - 如果 `vec1` 、 `vec2` 不是一个一维Tensor。
