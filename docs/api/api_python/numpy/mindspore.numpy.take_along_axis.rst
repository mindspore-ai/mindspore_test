mindspore.numpy.take_along_axis
=================================

.. py:function:: mindspore.numpy.take_along_axis(arr, indices, axis)

    根据一维索引和数据切片从输入数组中提取值。
    该函数沿指定的轴在索引和数据数组中迭代匹配一维切片，并使用前者在后者中查找值。这些切片可以具有不同的长度。

    参数：
        - **arr** (Tensor) - 源数组，shape为 ``(Ni…, M, Nk…)`` 。
        - **indices** (Tensor) - shape为 ``(Ni…, J, Nk…)`` 的索引，用于沿 ``arr`` 的每个一维切片取值。必须与 ``arr`` 的维度匹配，但维度 ``Ni`` 和 ``Nj`` 只需要与 ``arr`` 进行广播。
        - **axis** (int) - 沿该轴进行一维切片取值。如果 ``axis`` 为None，则输入数组将被视作首先被展平为一维。

    返回：
        Tensor，索引结果，shape为 ``(Ni…, J, Nk…)`` 。

    异常：
        - **ValueError** - 如果输入数组和索引的维度数量不同。
        - **TypeError** - 如果输入不是Tensor。