mindspore.hal.device_count
============================

.. py:function:: mindspore.hal.device_count(device_target=None)

    查询目标设备的数量，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.cpu.device_count` 、 :func:`mindspore.device_context.gpu.device_count` 、 :func:`mindspore.device_context.ascend.device_count` 代替。

    .. note::
        - 对于 ``"CPU"`` 设备，固定返回1。

    参数：
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        int
