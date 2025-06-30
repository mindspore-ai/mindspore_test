mindspore.numpy.fix
===================

.. py:function:: mindspore.numpy.fix(x)

    舍入至最接近零的相邻整数。

    将浮点数数组每个元素舍入为最接近零的相邻整数。舍入后的值以浮点数格式返回。

    .. note::
        不支持NumPy参数 `out` 。

    参数：
        - **x** (Tensor) - 需要舍入的float数组。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入不是Tensor。