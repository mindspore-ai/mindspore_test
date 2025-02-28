mindspore.runtime.max_memory_allocated
=======================================

.. py:function:: mindspore.runtime.max_memory_allocated()

    返回从进程启动开始，内存池实际被Tensor占用的内存大小的峰值。

    .. note::
        - 对于 `CPU` 硬件，固定返回0。

    返回：
        int，单位为Byte。
