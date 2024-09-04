mindspore.numpy.polyder
=======================

.. py:function:: mindspore.numpy.polyder(p, m=1)

    返回多项式指定阶数的导数。

    .. note::
        目前不支持NumPy对象poly1d。

    参数：
        - **p** (Union[int, float, bool, list, tuple, Tensor]) - 需要求导数的多项式。 一个代表多项式系数的序列。
        - **m** (int, 可选) - 默认值： ``1`` ，导数的阶数。

    返回：
        Tensor，表示导数的新多项式。

    异常：
        - **ValueError** - 如果 `p` 的维数超过1。