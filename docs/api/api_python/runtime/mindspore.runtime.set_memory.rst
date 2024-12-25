mindspore.runtime.set_memory
=============================

.. py:function:: mindspore.runtime.set_memory(init_size="2GB", increase_size="2GB", max_size="1024GB", optimize_level="O0")

    设置使用内存池实现的运行时设备内存管理的内存参数。

    框架默认设置所有参数，如下所示。

    参数：
        - **init_size** (str) - 内存池初始大小。格式为 ``xxGB``，默认为 ``2GB``。
        - **increase_size** (str) - 内存池的大小增加。当当前内存池没有
          足够的内存的时候，内存池将按此值扩展。格式为 ``xxGB``，默认为 ``2GB``。
        - **max_size** (str) - 内存池可用的最大内存。
          实际使用的内存大小是设备和最大设备内存的可用内存的最小值。
          格式为 ``xxGB``, 默认为设备的最大可用内存，表示为 ``1024GB``。
        - **optimize_level** (str) - 内存优化级别。该值必须在 [``O0``, ``O1``] 中，默认值： ``O0``。
