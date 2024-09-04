mindspore.numpy.polyadd
=======================

.. py:function:: mindspore.numpy.polyadd(a1, a2)

    找到两个多项式的和。 返回两个输入多项式的和所得的多项式。

    .. note::
        目前不支持NumPy对象poly1d。

    参数：
        - **a1** (Union[int, float, list, tuple, Tensor]) - 输入多项式。
        - **a2** (Union[int, float, list, tuple, Tensor]) - 输入多项式。

    返回：
        Tensor，输入之和。

    异常：
        - **ValueError** - 如果输入数组的维数超过1。