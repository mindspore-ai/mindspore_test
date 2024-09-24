mindspore.numpy.argmax
======================

.. py:function:: mindspore.numpy.argmax(a, axis=None)

    返回沿指定轴最大值的索引。

    .. note::
        不支持NumPy参数 `out` 。 在Ascend上，如果存在多个最大值的情况，返回的索引可能不一定对应于第一次出现的值。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。
        - **axis** (int, 可选) - 默认情况下，索引进入展平的数组，否则沿指定 `axis` 。 默认值： ``None`` 。

    返回：
        Tensor，原数组中元素的索引的数组。 与移除指定 `axis` 后的入参 `a` 的shape相同。

    异常：
        - **ValueError** - 如果 `axis` 超出范围。