mindspore.numpy.absolute
=================================

.. py:function:: mindspore.numpy.absolute(x, dtype=None)

    逐元素计算绝对值。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。目前后端内核仅支持浮点数计算，如果输入不是 ``float`` ，则会被转换为 ``mstype.float32`` 类型并再转换回来。

    参数：
        - **x** (Tensor) - 用于计算的Tensor。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。用于覆盖输出Tensor的dtype。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。