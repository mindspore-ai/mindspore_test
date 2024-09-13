mindspore.numpy.arcsin
=================================

.. py:function:: mindspore.numpy.arcsin(x, dtype=None)

    逐元素计算反正弦函数。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **x** (Tensor) - 输入Tensor，表示单位圆上的y坐标。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        输出Tensor。

    异常：
        - **TypeError** - 如果输入不是Tensor。