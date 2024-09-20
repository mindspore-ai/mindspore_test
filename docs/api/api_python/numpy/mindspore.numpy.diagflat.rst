mindspore.numpy.diagflat
=================================

.. py:function:: mindspore.numpy.diagflat(v, k=0)

    返回一个二维数组，其数组输入作为新输出数组的对角线。

    .. note::
        在GPU上，支持的数据类型是 ``np.float16`` 和 ``np.float32``。

    参数：
        - **v** (Tensor) - 输入数据，将其平坦化并设置为输出的第 ``k`` 个对角线。
        - **k** (int, 可选) - 需要的对角线；默认值： ``0`` 即主对角线， :math:`k>0` 意味着对角线高于主对角线，反之亦然。

    返回：
        2-D Tensor。

    异常：
        - **TypeError** - 如果输入不是一个Tensor。