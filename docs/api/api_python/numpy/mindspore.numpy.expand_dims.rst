mindspore.numpy.expand_dims
=================================

.. py:function:: mindspore.numpy.expand_dims(a, axis)

    扩展Tensor的shape。在扩展后的Tensor的shape中插入一个新轴，该轴将出现在指定的位置。

    参数：
        - **a** (Tensor) - 输入Tensor数组。
        - **axis** (Union[int, list(int), tuple(int)]) - 新轴在扩展轴中的位置。

    返回：
        Tensor，指定轴位置的维度数增加后的Tensor。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 ``axis`` 超出了 ``a.ndim`` 。