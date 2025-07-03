mindspore.hal.get_device_name
=============================

.. py:function:: mindspore.hal.get_device_name(device_id, device_target=None)

    返回指定卡号设备的设备名称，此接口将在后续版本中废弃。

    .. note::
        - 对于CPU设备，总是返回 ``"CPU"`` 。


    参数：
        - **device_id** (int) - 设备id。
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        str
