mindspore.numpy.triu_indices_from
=================================

.. py:function:: mindspore.numpy.triu_indices_from(arr, k=0)

    返回 ``arr`` 上三角的索引。

    参数：
        - **arr** (Union[Tensor, list, tuple]) – 二维数组。
        - **k** (int, 可选) – 对角线的偏移量，默认值： ``0`` 。

    返回：
        ``triu_indices_from`` ，由两个Tensor组成的的Tuple， ``arr`` 的上三角形的索引。

    异常：
        - **TypeError** – 如果 ``arr`` 不能转换为一个Tensor，或者 ``k`` 不是一个数字。
        - **ValueError** – 如果 ``arr`` 不能转换为一个二维的Tensor。