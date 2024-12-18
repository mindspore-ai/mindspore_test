mindspore.Tensor.addbmm
=======================

.. py:method:: mindspore.Tensor.addbmm(batch1, batch2, *, beta=1, alpha=1)

    详情请参考 :func:`mindspore.ops.addbmm`。

    .. py:method:: mindspore.Tensor.addbmm(batch1, batch2, *, beta=1, alpha=1)
        :noindex:

    对 `batch1` 和 `batch2` 应用批量矩阵乘法后进行reduced add， `self` 和最终的结果相加。
    `alpha` 和 `beta` 分别是 `batch1` 和 `batch2` 矩阵乘法和 `self` 的乘数。如果 `beta` 是0，那么 `self` 将会被忽略。

    .. math::
        output = \beta self + \alpha (\sum_{i=0}^{b-1} {batch1_i @ batch2_i})

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **batch1** (Tensor) - 矩阵乘法中的第一个Tensor。
        - **batch2** (Tensor) - 矩阵乘法中的第二个Tensor。

    关键字参数：
        - **beta** (Union[int, float]，可选) - `self` 的乘数。默认值： ``1`` 。
        - **alpha** (Union[int, float]，可选) - `batch1` @ `batch2` 的乘数。默认值： ``1`` 。

    返回：
        Tensor，和 `self` 具有相同的dtype。

    异常：
        - **TypeError** - 如果 `alpha`， `beta` 不是int或者float。
        - **ValueError** - 如果 `batch1`， `batch2` 不能进行批量矩阵乘法。
        - **ValueError** - 如果 `batch1` 或 `batch2` 不是三维Tensor。
