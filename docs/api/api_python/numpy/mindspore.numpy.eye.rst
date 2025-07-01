mindspore.numpy.eye
=================================

.. py:function:: mindspore.numpy.eye(N, M=None, k=0, dtype=mstype.float32)

    返回一个对角线上值为1，其他位置为0的二维Tensor。

    参数：
        - **N** (int) - 输出二维Tensor的行数。 输入值必须大于0。
        - **M** (int, 可选) - 输出二维Tensor的列数。默认值： ``None`` ，如果输入为 ``None`` ，则默认值： ``N`` ，输入值必须大于0。
        - **k** (int, 可选) - 对角线的索引。默认值： ``0`` ，即主对角线。正值表示上对角线，负值表示下对角线。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor ``dtype`` 。默认值： ``mstype.float32`` 。

    返回：
        Tensor，shape为(N, M)。其中，除第 ``k`` 个对角线值等于1外，其余所有元素都等于0。

    异常：
        - **TypeError** - 如果输入参数非给定的数据类型。