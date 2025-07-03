mindspore.Tensor.argmin_with_value
===================================

.. py:method:: mindspore.Tensor.argmin_with_value(axis=0, keep_dims=False)

    返回tensor在指定轴上的最小值及其索引。

    参数：
        - **axis** (Union[int, None], 可选) - 指定计算轴。如果为 ``None`` ，计算tensor中的所有元素。默认 ``0`` 。
        - **keep_dims** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        两个tensor组成的tuple(min, min_indices)。
