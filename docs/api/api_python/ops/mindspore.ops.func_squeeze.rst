mindspore.ops.squeeze
=====================

.. py:function:: mindspore.ops.squeeze(input, axis=None)

    删除输入tensor中长度为1的轴。

    .. note::
        - 请注意，在动态图模式下，输出tensor将与输入tensor共享数据，并且没有tensor数据复制过程。
        - 维度索引从0开始，并且必须在 `[-input.ndim, input.ndim)` 范围内。
        - 在GE模式下，只支持对input shape中大小为1的维度进行压缩。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int)]) - 待删除的轴，默认 ``None`` 。

    返回：
        Tensor
