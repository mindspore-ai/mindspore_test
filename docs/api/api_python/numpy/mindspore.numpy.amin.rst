mindspore.numpy.amin
=================================

.. py:function:: mindspore.numpy.amin(a, axis=None, keepdims=False, initial=None, where=True)

    返回数组的最小值或沿指定轴的最小值。

    .. note::
        不支持Numpy的 ``out`` 参数。在GPU上，支持的数据类型为 ``np.float16`` 和 ``np.float32``。

    参数：
        - **a** (Tensor) - 输入数据。
        - **axis** (Union[int, tuple(int), None], 可选) - 默认值： ``None`` 。指定操作的轴或多个轴。默认情况下，将使用展平后的输入。如果该参数是整数组成的tuple，则会在多个轴上选择最小值，而不是像之前那样选择单个轴或所有轴上的最小值。
        - **keepdims** (bool, 可选) - 默认值： ``False`` 。如果设置为 ``True`` ，则保留被缩减的轴，作为结果中大小为一的维度。使用此选项，结果将与输入数组正确广播。
        - **initial** (Number, 可选) - 默认值： ``None`` 。输出元素的最大值。必须存在才能在空切片上进行计算。
        - **where** (bool Tensor, 可选) - 默认值： ``True`` 。一个布尔数组，被广播以匹配数组的维度，并选择包含在计算中的元素。如果传递了非默认值，则 ``initial`` 也必须提供。

    返回：
        Tensor或标量， ``a`` 的最小值。如果 ``axis`` 为 ``None`` ，则结果是一个标量值。如果给定 ``axis`` ，则结果是一个维度为 ``a.ndim - 1`` 的数组。

    异常：
        - **TypeError** - 如果输入不是Tensor。