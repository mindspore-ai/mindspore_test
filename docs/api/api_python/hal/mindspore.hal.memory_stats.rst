mindspore.hal.memory_stats
==========================

.. py:function:: mindspore.hal.memory_stats(device_target=None)

    返回从内存池查询到的状态信息，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.memory_stats` 代替。

    .. note::
        - 对于  `CPU` 设备，固定返回数据为空的字典。

    参数：
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        dict，查询到的内存信息。
