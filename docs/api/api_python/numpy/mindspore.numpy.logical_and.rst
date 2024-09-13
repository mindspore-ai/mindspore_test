mindspore.numpy.logical_and
=================================

.. py:function:: mindspore.numpy.logical_and(x1, x2, dtype=None)

    逐元素计算 ``x1`` 和 ``x2`` 的逻辑与（AND）的真值。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **x1** (Tensor) - 输入Tensor。
        - **x2** (Tensor) - 输入Tensor。如果 :math:`x1.shape != x2.shape` ，则它们必须能够广播到一个共同的shape（该shape成为输出的shape）。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。对 ``x1`` 和 ``x2`` 中的元素执行逻辑与（AND）操作的Boolean结果；shape由广播确定。如果 ``x1`` 和 ``x2`` 都是标量，则返回标量。