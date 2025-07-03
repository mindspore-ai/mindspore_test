mindspore.runtime.memory_allocated
===================================

.. py:function:: mindspore.runtime.memory_allocated()

    返回当前实际被Tensor占用的内存大小。

    .. note::
        - 对于 `CPU` 硬件，固定返回0。

    返回：
        int，单位为Byte。
