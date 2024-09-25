mindspore.numpy.cumsum
======================

.. py:function:: mindspore.numpy.cumsum(a, axis=None, dtype=None)

    返回沿给定 `axis` 的元素的累计和。

    .. note::
        如果 `a.dtype` 为 `int8` 、 `int16` 或 `bool` ，结果的 `dtype` 将提升至 `int32` 。

    参数：
        - **a** (Tensor) - 输入Tensor。
        - **axis** (int, 可选) - 计算累计和所沿的轴。 默认为 `None` ，计算被展开的数组的累计和。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果未指定，保持与输入Tensor `a` 相同，除非 `a` 的整数精度低于默认平台整数的精度。 在这种情况下，使用默认平台整数类型。 默认值： ``None`` 。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入参数的类型不是上述指定类型。
        - **ValueError** - 如果 `axis` 超出范围。