mindspore.numpy.polymul
=========================

.. py:function:: mindspore.numpy.polymul(a1, a2)

    求两个多项式的积。

    .. note::
        目前不支持NumPy对象poly1d。

    参数：
        - **a1** (Union[int, float, bool, list, tuple, Tensor]) - 输入多项式。
        - **a2** (Union[int, float, bool, list, tuple, Tensor]) - 输入多项式。

    返回：
        Tensor，表示导数的新多项式。

    异常：
        - **ValueError** - 如果输入数组的维数超过1。