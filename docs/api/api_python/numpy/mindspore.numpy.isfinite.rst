mindspore.numpy.isfinite
=================================

.. py:function:: mindspore.numpy.isfinite(x, dtype=None)

    逐元素测试是否为有限数（不是无穷大或非数值）。
    结果将以bool数组的形式返回。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。在GPU上，支持的dtype为 ``np.float16`` 和 ``np.float32``。

    参数：
        - **x** (Tensor) - 输入值。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，如果 ``x`` 不是正无穷大、负无穷大或NaN，则返回为True；否则为False。如果 ``x`` 是标量，则返回标量。