mindspore.hal.memory_summary
============================

.. py:function:: mindspore.hal.memory_summary(device_target=None)

    返回可读的内存池状态信息，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.memory_summary` 代替。

    参数：
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        str，表格形式的可读内存池状态信息。
