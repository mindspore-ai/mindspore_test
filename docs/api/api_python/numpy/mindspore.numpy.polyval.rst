mindspore.numpy.polyval
=======================

.. py:function:: mindspore.numpy.polyval(p, x)

    在特定值处求多项式的值。如果 `p` 的长度为 `N` ，该函数返回值为：

    ``p[0]x*(N-1) + p[1]x*(N-2) + ... + p[N-2]*x + p[N-1]`` 如果 `x` 是一个序列，那么返回 ``p(x)`` 对应 `x` 中每个元素的值。如果 `x` 是另一个多项式，则返回复合多项式 `p(x(t))` 。

    .. note::
        目前不支持NumPy对象poly1d。

    参数：
        - **p** (Union[int, float, bool, list, tuple, Tensor]) - 从最高次项到常数项的一维多项式系数数组(包括等于零的系数)。
        - **x** (Union[int, float, bool, list, tuple, Tensor]) - 求对应 `p` 值的数或数组。

    返回：
        Tensor。

    异常：
        - **ValueError** - 如果 `p` 的维数超过1。

