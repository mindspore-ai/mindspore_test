mindspore.numpy.apply_along_axis
=================================

.. py:function:: mindspore.numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    在指定轴的一维切片上调用给定函数。执行 ``func1d(a, *args, **kwargs)`` ，其中 ``func1d`` 在一维数组上运算， ``a`` 是 ``arr`` 沿着指定 ``axis`` 的一维切片。

    参数：
        - **func1d** (function) - Maps  ``(M,) -> (Nj…)`` 。该函数仅接受输入为一维数组，应用于沿指定轴的 ``arr`` 的一维切片。
        - **axis** (int) - 指定 ``arr`` 所需切片的轴。
        - **arr** (Tensor) - 输入的数组，且输入包含数组shape ``(Ni…, M, Nk…)`` 。
        - **args** (any) -  ``func1d`` 的附加参数。
        - **kwargs** (any) -  ``func1d`` 的附加命名参数。

    返回：
        Tensor，shape为 ``(Ni…, Nj…, Nk…)`` ，除了 ``axis`` 那一维 ，它的shape与 ``arr`` 的shape相同。 ``axis`` 那一维被替换为 ``func1d`` 的返回值shape。因此，如果 ``func1d`` 返回标量，则输出的维度将比 ``arr`` 少一个。

    异常：
        - **ValueError** - 如果 ``axis`` 超出索引范围。