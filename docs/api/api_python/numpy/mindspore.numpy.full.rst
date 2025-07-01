mindspore.numpy.full
=================================

.. py:function:: mindspore.numpy.full(shape, fill_value, dtype=None)

    返回一个给定shape、类型，并用 ``fill_value`` 填充的新数组。

    参数：
        - **shape** (Union[int, tuple(int), list(int)]) - 指定的Tensor的shape，例如： :math:`(2, 3)` 或 :math:`2` 。
        - **fill_value** (Union[int, float, bool, list, tuple]) - 填充值，可以为标量或数组。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor ``dtype`` 。如果 ``dtype`` 为 ``None`` ，则将从 ``fill_value`` 推断出新Tensor的数据类型。默认值： ``None`` 。

    返回：
        Tensor，具有给定的shape、类型，并用 ``fill_value`` 填充。

    异常：
        - **TypeError** - 如果输入参数非给定的数据类型。
        - **ValueError** - 如果 ``shape`` 的元素数目小于0。