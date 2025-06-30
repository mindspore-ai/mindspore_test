mindspore.ops.mvlgamma
=======================

.. py:function:: mindspore.ops.mvlgamma(input, p)

    逐元素计算 `p` 维多元对数伽马函数值。

    Mvlgamma计算公式如下：

    .. math::

        \log (\Gamma_{p}(input))=C+\sum_{i=1}^{p} \log (\Gamma(input-\frac{i-1}{2}))

    其中 :math:`C = \log(\pi) \times \frac{p(p-1)}{4}` ，:math:`\Gamma(\cdot)` 为Gamma函数。

    参数：
        - **input** (Tensor) - 多元对数伽马函数的输入tensor。`input` 中每个元素的值必须大于 :math:`(p - 1) / 2` 。
        - **p** (int) - 维度数量，必须大于等于1。

    返回：
        Tensor
