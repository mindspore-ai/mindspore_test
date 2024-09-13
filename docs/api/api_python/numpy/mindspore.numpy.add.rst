mindspore.numpy.add
=================================

.. py:function:: mindspore.numpy.add(x1, x2, dtype=None)

    逐元素相加两个参数。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **x1** (Tensor) - 参与相加的第一个输入。
        - **x2** (Tensor) - 参与相加的第二个输入。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。用于覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， ``x1`` 和 ``x2`` 逐元素相加后的和。如果 ``x1`` 和 ``x2`` 都是标量，则返回标量。