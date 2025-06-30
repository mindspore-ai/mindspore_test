mindspore.numpy.full_like
=================================

.. py:function:: mindspore.numpy.full_like(a, fill_value, dtype=None, shape=None)

    返回一个与给定数组具有相同shape和类型的完整数组。

    .. note::
        - 输入数组在一个维度上的大小必须相同。
        - 如果 `a` 不是Tensor，`dtype` 会默认为float32。

    参数：
        - **a** (Union[Tensor, list, tuple]) - 原数组，返回数组的shape和数据类型与 ``a`` 相同。
        - **fill_value** (scalar) - 填充值。
        - **dtype** (mindspore.dtype, 可选) - 覆盖结果的数据类型。
        - **shape** (int, ints的序列, 可选) - 覆盖结果的shape。

    返回：
        Tensor，与 ``a`` 的shape、类型相同，并用 ``fill_value`` 填充。

    异常：
        - **ValueError** - 如果 ``a`` 的数据类型不是Tensor、List或Tuple。