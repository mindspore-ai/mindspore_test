mindspore.numpy.copy
=================================

.. py:function:: mindspore.numpy.copy(a)

    返回给定对象的Tensor副本。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 可以转换为Tensor的任何形式的输入数据。 包含 ``int, float, bool, Tensor, list, tuple`` 。

    返回：
        Tensor，有与 ``a`` 相同的数据。

    异常：
        - **TypeError** - 如果没有按照上述内容给定的数据类型输入参数。
        - **ValueError** - 如果输入 ``a`` 在不同的维度上具有不同的大小。