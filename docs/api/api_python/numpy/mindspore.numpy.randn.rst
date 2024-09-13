mindspore.numpy.randn
=================================

.. py:function:: mindspore.numpy.randn(*shape, dtype=mstype.float32)

    返回一个给定shape和类型的新Tensor，并填充来自标准正态分布中的一个（或多个）样本。

    参数：
        - **shape** (Union[int, tuple(int), list(int)]) - 指定新Tensor的shape，例如 :math:`(2,3)` 或 :math:`2` 。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor数据类型，必须是float型数据。默认值： ``mstype.float32`` 。

    返回：
        Tensor，给定shape和类型，并填充来自“标准正态”分布中的一个（或多个）样本。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 ``dtype`` 不是float类型。
