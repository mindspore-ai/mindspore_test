mindspore.numpy.logical_or
=================================

.. py:function:: mindspore.numpy.logical_or(x1, x2, dtype=None)

    逐元素计算 ``x1`` 和 ``x2`` 的逻辑或（OR）的真值。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **x1** (Tensor) – 输入Tensor。
        - **x2** (Tensor) – 输入Tensor。如果 :math:`x1.shape != x2.shape` ，则它们必须能够广播到一个共同的shape（该shape成为输出的shape）。
        - **dtype** (mindspore.dtype, 可选) – 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， ``x1`` 和 ``x2`` 的逐元素逻辑或（OR）的比较结果。通常为bool类型，除非传入 ``dtype=object`` 。如果 ``x1`` 和 ``x2`` 都是标量，则返回标量。