mindspore.ops.eye
==================

.. py:function:: mindspore.ops.eye(n, m=None, dtype=None)

    返回一个主对角线上元素为1，其余元素为0的tensor。

    参数：
        - **n** (int) - 返回的行数。
        - **m** (int，可选) - 返回的列数。如果为None，与n相等。默认 ``None`` 。
        - **dtype** (mindspore.dtype，可选) - 指定数据类型。默认 ``None`` 。
    返回：
        Tensor