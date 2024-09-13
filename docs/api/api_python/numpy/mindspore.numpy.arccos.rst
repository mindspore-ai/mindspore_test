mindspore.numpy.arccos
=================================

.. py:function:: mindspore.numpy.arccos(input, dtype=None)

    逐元素计算反余弦函数。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **input** (Tensor) - 输入Tensor，表示单位圆上的x坐标。对于实数输入，定义域为 :math:`[-1, 1]` 。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入不是Tensor。