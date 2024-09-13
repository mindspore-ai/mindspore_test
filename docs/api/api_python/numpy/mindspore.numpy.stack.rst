mindspore.numpy.stack
=================================

.. py:function:: mindspore.numpy.stack(arrays, axis=0)

    沿新轴连接一系列数组。 ``axis`` 参数指定结果中新轴的索引。例如，如果 ``axis=0`` ，它将成为第一个维度；如果 ``axis=-1`` ，它将成为最后一个维度。

    .. note::
        不支持Numpy的 ``out`` 参数。

    参数：
        - **arrays** (Tensor的序列) - 每个数组必须具有相同的shape。
        - **axis** (int, 可选) - 结果数组中沿输入数组堆叠的轴。默认值： ``0`` 。

    返回：
        Tensor，堆叠后的数组比输入数组多一个维度。

    异常：
        - **ValueError** - 如果输入不是Tensor、tuple或list。