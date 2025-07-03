mindspore.numpy.interp
======================

.. py:function:: mindspore.numpy.interp(x, xp, fp, left=None, right=None)

    用于单调递增的样本点的一维线性插值。 返回包含给定离散数据点(xp, fp)的函数进行一维分段线性插值后x处的函数值。

    .. note::
        Numpy参数period不受支持。 不支持复数值。

    参数：
        - **x** (Union[int, float, bool, list, tuple, Tensor]) - 计算插值后的值时使用的x坐标。
        - **xp** (Union[int, float, bool, list, tuple, Tensor]) - 元素为float的1-D序列，输入数据点的x坐标，必须递增。
        - **fp** (Union[int, float, bool, list, tuple, Tensor]) - 元素为float的1-D序列，输入数据点的y坐标，与 `xp` 等长。
        - **left** (float, 可选) - ``x < xp[0]`` 时返回的值，一旦存在，默认值为 ``fp[0]`` 。默认值： ``None`` 。
        - **right** (float, 可选) - ``x > xp[-1]`` 时返回的值，一旦存在，默认值为 ``fp[-1]`` 。默认值： ``None`` 。

    返回：
        Tensor，插值得到的值，其shape与 `x` 相同。

    异常：
        - **ValueError** - 如果 `xp` 或 `fp` 不是一维的，或如果 `xp` 和 `fp` 的长度不同。
    