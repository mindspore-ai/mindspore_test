mindspore.numpy.squeeze
=================================

.. py:function:: mindspore.numpy.squeeze(a, axis=None)

    从Tensor的shape中移除单维元素。

    参数：
        - **a** (Tensor) – 输入Tensor数组。
        - **axis** (Union[None, int, list(int), tuple(list)]) – 要压缩的轴，默认值: ``None`` 。

    返回：
        Tensor，移除了所有或部分长度为1的维度。

    异常：
        - **TypeError** – 如果输入参数非上述给定的类型。
        - **ValueError** – 如果指定的轴具有 :math:`>1` 的shape元素。