mindspore.Tensor.type_as
========================

.. py:method:: Tensor.type_as(other)

    将第一个输入的Tensor的数据类型转换为第二个输入的Tensor的数据类型。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        将复数转换为bool类型的时候，不考虑复数的虚部，只要实部不为零，返回True，否则返回False。

    参数：
        - **tensor** (Tensor) - 数据类型为指定类型的Tensor，其shape为 :math:`(x_0, x_1, ..., x_R)` 。

    返回：
        Tensor，其shape与输入Tensor相同，即 :math:`(x_0, x_1, ..., x_R)` 。

    异常：
        - **TypeError** - `tensor` 不是Tensor。
