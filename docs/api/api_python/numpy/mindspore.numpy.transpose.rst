mindspore.numpy.transpose
=================================

.. py:function:: mindspore.numpy.transpose(a, axes=None)

    反转或交换Tensor的轴；返回修改后的Tensor。

    参数：
        - **a** (Tensor) - 要转置的Tensor。
        - **axes** (Union[None, tuple, list]，可选) - 轴的顺序。如果 ``axes`` 为 ``None`` ，则转置整个Tensor。默认值： ``None`` 。

    返回：
        Tensor，转置后的Tensor数组。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 ``axes`` 的数量不等于 ``a.ndim`` 。