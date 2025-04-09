mindspore.hal.get_device_capability
===================================

.. py:function:: mindspore.hal.get_device_capability(device_id, device_target=None)

    返回指定卡号设备的设备能力，此接口将在后续版本中废弃。

    参数：
        - **device_id** (int) - 设备id。
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        对于GPU设备，返回tuple(param1, param2)。

        - param1：int，cuda major版本号。
        - param2：int，cuda minor版本号。

        对于CPU以及Ascend设备，返回None。
