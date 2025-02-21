mindspore.mint.baddbmm
======================

.. py:function:: mindspore.mint.baddbmm(input, batch1, batch2, *, beta=1, alpha=1)

    对输入的两个三维矩阵batch1与batch2相乘，并将结果与input相加。
    计算公式定义如下：

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    参数：
        - **input** (Tensor) - 输入Tensor。当batch1是 :math:`(C, W, T)` 的Tensor而且batch2是一个 :math:`(C, T, H)` 的Tensor时，输入必须为可以被广播为 :math:`(C, W, H)` 形状的Tensor。
        - **batch1** (Tensor) - 公式中的 :math:`batch1` 。必须为3-D的Tensor，类型与 `input` 一致。
        - **batch2** (Tensor) - 公式中的 :math:`batch2` 。必须为3-D的Tensor，类型与 `input` 一致。

    关键字参数：
        - **beta** (Union[float, int], 可选) - 输入的乘数。默认值： ``1`` 。
        - **alpha** (Union[float, int]，可选) - :math:`batch1 @ batch2` 的系数，默认值： ``1`` 。

    返回：
        Tensor，其数据类型与 `input` 相同，维度为 :math:`(C, W, H)`。

    异常：
        - **TypeError** - `input` 、 `batch1` 或 `batch2` 的类型不是Tensor。
        - **TypeError** - `input` 、 `batch1` 或 `batch2` 数据类型不一致。
        - **ValueError** - `batch1` 或 `batch2` 不是三维Tensor。
