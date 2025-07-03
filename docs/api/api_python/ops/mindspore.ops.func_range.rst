mindspore.ops.range
====================

.. py:function:: mindspore.ops.range(start, end, step, maxlen=1000000)

    返回一个在 [ `start`, `end` ) 区间内，步长为 `step` 的tensor。

    .. note::
        三个输入必须全为整数或全为浮点数。

    参数：
        - **start** (number) - 区间的起始值。
        - **end** (number) - 区间的末尾数。
        - **step** (number) - 值的间隔。
        - **maxlen** (int，可选) - 该算子将会被分配能够存储 `maxlen` 个数据的内存。该参数是可选的，必须为正数，默认 ``1000000`` 。如果输出的数量超过 `maxlen` ，将会引起运行时错误。

    返回：
        Tensor
