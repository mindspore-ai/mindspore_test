mindspore.numpy.squeeze
=================================

.. py:function:: mindspore.numpy.squeeze(a, axis=None)

    返回删除指定 `axis` 中大小为1的维度后的Tensor。

    如果 :math:`axis=None` ，则删除所有大小为1的维度。
    如果指定了 `axis`，则删除指定 `axis` 中大小为1的维度。
    例如，如果不指定维度 :math:`axis=None` ，输入的shape为(A, 1, B, C, 1, D)，则输出的Tensor的shape为(A, B, C, D)。如果指定维度，squeeze操作仅在指定维度中进行。
    如果输入的shape为(A, 1, B)， :math:`axis=0` 或 :math:`axis=2` 时不会改变输入的Tensor，但 :math:`axis=1` 时会使输入Tensor的shape变为(A, B)。

    参数：
        - **a** (Tensor) - 输入Tensor数组。
        - **axis** (Union[None, int, list(int), tuple(list)]，可选) - 要压缩的轴，默认值: ``None`` 。

    返回：
        Tensor，移除了所有或部分长度为1的维度。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果指定的轴具有 :math:`>1` 的shape元素。