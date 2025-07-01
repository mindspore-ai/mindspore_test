mindspore.numpy.kron
====================

.. py:function:: mindspore.numpy.kron(a, b)

    两个数组的Kronecker积。

    计算Kronecker积，即先将第二个数组分块，然后使用第一个数组缩放后组成的复合数组。

    .. note::
        不支持bool值。

    参数：
        - **a** (Union[int, float, list, tuple, Tensor]) - 输入值。
        - **b** (Union[int, float, list, tuple, Tensor]) - 输入值。

    返回：
        Tensor。
