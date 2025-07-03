mindspore.numpy.rint
====================

.. py:function:: mindspore.numpy.rint(x, dtype=None)

    将数组的元素四舍五入到最接近的整数。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。

    参数：
        - **x** (Union[float, list, tuple, Tensor]) - 任意维度的输入Tensor。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        输出Tensor的shape和类型与 `x` 相同。如果 `x` 是标量，则返回标量。

    异常：
        - **TypeError** - 如果 `x` 无法转换为Tensor。
    