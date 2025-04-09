mindspore.hal.reset_max_memory_allocated
========================================

.. py:function:: mindspore.hal.reset_max_memory_allocated(device_target=None)

    重置内存池真实被tensor占用的内存大小的峰值，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.reset_max_memory_allocated` 代替。

    参数：
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。
