mindspore.numpy.signbit
=================================

.. py:function:: mindspore.numpy.signbit(x, dtype=None)

    逐元素扫描元素的符号位，如果符号位为1（即元素小于0）则返回True。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **x** (Union[int, float, bool, list, tuple, Tensor]) - 输入值。
        - **dtype** (mindspore.dtype, 可选) - 默认值: ``None`` 。覆盖输出 Tensor的dtype。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入不是类似数组的对象，或者 ``dtype`` 不是 ``None`` 或 ``bool`` 。