mindspore.Tensor.addmm
======================

.. py:method:: mindspore.Tensor.addmm(mat1, mat2, *, beta=1, alpha=1)

    详情请参考 :func:`mindspore.ops.addmm`。

    .. py:method:: mindspore.Tensor.addmm(mat1, mat2, *, beta=1, alpha=1)
        :noindex:

    对输入的两个二维矩阵mat1与mat2相乘，并将结果与self相加。
    计算公式定义如下：

    .. math::
        output = \beta self + \alpha (mat1 @ mat2)

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **mat1** (Tensor) - 第一个待乘矩阵，必须为2-D的Tensor，类型与 `self` 一致。
        - **mat2** (Tensor) - 第二个待乘矩阵，必须为2-D的Tensor，类型与 `self` 一致。

    关键字参数：
        - **beta** (Union[float, int], 可选) - 输入的乘数。默认值： ``1`` 。
        - **alpha** (Union[float, int]，可选) - :math:`mat1 @ mat2` 的系数，默认值： ``1`` 。

    返回：
        Tensor，其数据类型与 `self` 相同，其shape和mat1 @ mat2相同。

    异常：
        - **TypeError** - `self` 、 `mat1` 或 `mat2` 的类型不是Tensor。
        - **TypeError** - `self` 、 `mat1` 或 `mat2` 数据类型不一致。
        - **ValueError** - `mat1` 或 `mat2` 的不是二维Tensor。
