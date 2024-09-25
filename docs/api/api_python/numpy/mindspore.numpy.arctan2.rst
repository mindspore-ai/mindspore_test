mindspore.numpy.arctan2
=======================

.. py:function:: mindspore.numpy.arctan2(x1, x2, dtype=None)

    逐元素计算 :math:`x1/x2` 的反正切，并正确选择象限。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x1** (Tensor) - 输入Tensor。
        - **x2** (Tensor) - 输入Tensor。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype 。

    返回：
        Tensor或标量， `x1` 与 `x2` 逐元素的计算结果。如果 `x1` 与 `x2` 都是标量，则结果也是标量。