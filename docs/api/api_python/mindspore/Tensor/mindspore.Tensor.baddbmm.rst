mindspore.Tensor.baddbmm
========================

.. py:method:: mindspore.Tensor.baddbmm(batch1, batch2, *, beta=1, alpha=1)

    对输入的两个三维矩阵batch1与batch2相乘，并将结果与self相加。
    计算公式定义如下：

    .. math::
        \text{out}_{i} = \beta \text{self}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    参数：
        - **batch1** (Tensor) - 公式中的 :math:`batch1` 。必须为3-D的Tensor，类型与 `self` 一致。
        - **batch2** (Tensor) - 公式中的 :math:`batch2` 。必须为3-D的Tensor，类型与 `self` 一致。

    关键字参数：
        - **beta** (Union[float, int], 可选) - 输入的乘数。默认值： ``1`` 。
        - **alpha** (Union[float, int]，可选) - :math:`batch1 @ batch2` 的系数，默认值： ``1`` 。

    返回：
        Tensor，其数据类型与 `self` 相同，维度为 :math:`(C, W, H)`。

    异常：
        - **TypeError** - `self` 、 `batch1` 或 `batch2` 的类型不是Tensor。
        - **TypeError** - `self` 、 `batch1` 或 `batch2` 数据类型不一致。
        - **ValueError** - `batch1` 或 `batch2` 的不是三维Tensor。
