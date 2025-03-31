mindspore.ops.chunk
====================

.. py:function:: mindspore.ops.chunk(input, chunks, axis=0)

    沿着指定轴将输入tensor切分成多个子tensor。

    .. note::
        此函数返回的数量可能小于通过 `chunks` 指定的数量!

    参数：
        - **input** (Tensor) - 被切分的tensor。
        - **chunks** (int) - 切分的数量。
        - **axis** (int，可选) - 切分的轴。默认 ``0`` 。

    返回：
        由tensor组成的tuple。
