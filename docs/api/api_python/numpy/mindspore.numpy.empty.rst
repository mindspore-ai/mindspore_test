mindspore.numpy.empty
=================================

.. py:function:: mindspore.numpy.empty(shape, dtype=mstype.float32)

    返回一个给定shape和类型的新数组，而不进行初始化。

    .. note::
        不支持原生Numpy接口的 ``order`` 参数，不支持对象数组。

    参数：
        - **shape** (Union[int, tuple(int)]) - 空数组的shape，例如：(2, 3) 或 2。
        - **dtype** (mindspore.dtype, 可选) - 数组所需输出的数据类型，例如 ``mstype.int8`` 。默认值： ``mstype.float32`` 。

    返回：
        Tensor，给定shape、数据类型的未初始化（任意）数据的数组。

    异常：
        - **TypeError** - 如果输入的shape或数据类型无效。