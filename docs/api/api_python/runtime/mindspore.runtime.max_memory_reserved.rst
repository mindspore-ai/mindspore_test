mindspore.runtime.max_memory_reserved
======================================

.. py:function:: mindspore.runtime.max_memory_reserved()

    返回从进程启动开始，内存池管理的内存总量的峰值。

    .. note::
        - 对于 `CPU` 硬件，固定返回0。

    返回：
        int，单位为Byte。
