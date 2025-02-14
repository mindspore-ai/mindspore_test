mindspore.Tensor.bmm
====================

.. py:method:: mindspore.Tensor.bmm(mat2)

    基于batch维度的两个三维Tensor的矩阵乘法。 `self` 必须是三维Tensor，shape为 :math:`(b, n, m)` 。

    .. math::
        \text{output} = \text{self} @ \text{mat2}

    参数：
        - **mat2** (Tensor) - 输入相乘的第二个Tensor。必须是三维Tensor，shape为 :math:`(b, m, p)` 。

    返回：
        Tensor，输出Tensor的shape为 :math:`(b, n, p)` 。其中每个矩阵是 `self` 的batch中相应矩阵的乘积。

    异常：
        - **ValueError** - `self` 或 `mat2` 的维度不为3。
        - **ValueError** - `self` 第三维的长度不等于 `mat2` 第二维的长度。
        - **ValueError** - `self` 的 batch 维长度不等于 `mat2` 的 batch 维长度。
