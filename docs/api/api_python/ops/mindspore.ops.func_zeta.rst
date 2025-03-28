mindspore.ops.zeta
===================

.. py:function:: mindspore.ops.zeta(input, other)

    逐元素计算Hurwitz zeta的值。

    .. math::

        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Union[Tensor, int, float]) - 第一个输入tensor。在公式中表示为 :math:`x` 。
        - **other** (Union[Tensor, int, float]) - 第二个输入tensor。在公式中表示为 :math:`q` 。

    返回：
        Tensor