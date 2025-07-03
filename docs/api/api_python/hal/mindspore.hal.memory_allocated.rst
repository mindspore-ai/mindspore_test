mindspore.hal.memory_allocated
==============================

.. py:function:: mindspore.hal.memory_allocated(device_target=None)

    返回当前真实被tensor占用的内存大小，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.memory_allocated` 代替。

    .. note::
        - 对于 `CPU` 设备，固定返回0。

    参数：
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        int，单位为Byte。
