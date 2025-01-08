mindspore.mint.repeat_interleave
================================

.. py:function:: mindspore.mint.repeat_interleave(input, repeats, dim=None, *, output_size=None)

    沿着轴重复Tensor的元素，类似 `numpy.repeat`。

    .. warning::
        仅Atlas A2训练系列产品支持。

    参数：
        - **input** (Tensor) - 进行重复操作的入参Tensor，类型必须为float16，float32，int8，uint8，int16，int32或者int64。
        - **repeats** (Union[int, tuple, list, Tensor]) - 指定复制次数，为正数。
        - **dim** (int, 可选) - 指定复制轴，默认值： ``None`` 。如果为 ``None`` ，输入Tensor会被展平并且输出结果也会被展平。

    关键字参数：
        - **output_size** (int, 可选) - 给定轴的总输出大小(即参数repeats元素之和)，默认值： ``None`` 。

    返回：
        Tensor，值沿指定轴复制。如果输入的shape为 :math:`(s1, s2, ..., sn)` ，轴为i，则输出的shape为 :math:`(s1, s2, ..., si * repeats, ..., sn)` 。输出的数据类型与输入相同。
