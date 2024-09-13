mindspore.numpy.identity
=================================

.. py:function:: mindspore.numpy.identity(n, dtype=mstype.float32)

    返回单位数组。主对角线上全为1。

    参数：
        - **n** (int) - 设置的输出数组的行数和列数，必须大于0。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor ``dtype`` ，默认值： ``mstype.float32`` 。

    返回：
        Tensor，shape为 ``(n, n)`` ，除主对角线上值为1，其余所有元素都等于0。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。