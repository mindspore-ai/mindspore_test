mindspore.numpy.polysub
=======================

.. py:function:: mindspore.numpy.polysub(a1, a2)

    两个多项式的差(减法)。 给定两个多项式 `a1` 和 `a2` ，返回 ``a1 - a2`` 。

    .. note::
        目前不支持NumPy对象poly1d。

    参数：
        - **a1** (Union[int, float, list, tuple, Tensor]) - 被减多项式。
        - **a2** (Union[int, float, list, tuple, Tensor]) - 减数多项式。

    返回：
        Tensor，输入的差。

    异常：
        - **ValueError** - 如果输入数组的维数超过1。