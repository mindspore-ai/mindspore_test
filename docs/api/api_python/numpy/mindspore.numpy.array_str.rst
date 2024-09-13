mindspore.numpy.array_str
=================================

.. py:function:: mindspore.numpy.array_str(a)

    返回数组中数据的字符串表示形式。
    数组中的数据以单个字符串的形式返回。此函数类似于 ``array_repr`` ，不同之处在于 ``array_repr`` 还返回数组的类型及其数据类型的信息。

    .. note::
        不支持Numpy的 ``max_line_width`` 、 ``precision`` 和 ``suppress_small`` 参数。在图模式下不支持该函数。

    参数：
        - **a** (Tensor) - 输入数据。

    返回：
        字符串。

    异常：
        - **TypeError** - 如果输入不是Tensor。