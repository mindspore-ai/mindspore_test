mindspore.Tensor.isinf
======================

.. py:method:: mindspore.Tensor.isinf()

    确定 `self` 每个位置上的元素是否为正无穷或负无穷。

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } self_{i} = \text{Inf} \\
          & \ False,\ \text{ if } self_{i} \ne  \text{Inf}
        \end{cases}

    其中 :math:`Inf` 表示无穷大。

    .. warning::
        - 该API目前只支持在Atlas A2训练系列产品上使用。

    返回：
        Tensor，shape与 `self` 相同，数据的类型为bool。
