mindspore.numpy.ediff1d
=======================

.. py:function:: mindspore.numpy.ediff1d(ary, to_end=None, to_begin=None)

    计算Tensor中连续元素之间的差值。

    参数：
        - **ary** (Tensor) - 如果需要，将在取差值之前被展平。
        - **to_end** (Tensor, scalar, 可选) - 在返回的差值末尾添加的数字。默认值： ``None`` 。
        - **to_begin** (Tensor, scalar, 可选) - 在返回的差值开头添加的数字。默认值： ``None`` 。

    返回：
        差值。

    异常：
        - **TypeError** - 如果输入类型不是上述指定的类型。