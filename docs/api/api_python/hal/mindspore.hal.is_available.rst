mindspore.hal.is_available
=============================

.. py:function:: mindspore.hal.is_available(device_target)

    查询目标设备是否可用，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.cpu.is_available` 、 :func:`mindspore.device_context.gpu.is_available` 、 :func:`mindspore.device_context.ascend.is_available` 代替。

    参数：
        - **device_target** (str) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。

    返回：
        bool
