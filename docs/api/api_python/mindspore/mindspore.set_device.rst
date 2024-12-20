mindspore.set_device
=====================

.. py:function:: mindspore.set_device(device_target, device_id=0)

    设置运行环境的设备目标和设备ID。

    .. note::
        - `device_target` 的取值必须在 ["CPU", "GPU", "Ascend"] ，没有默认值。
    
    参数：
        - **device_target** (str) - 要运行的目标设备，仅支持 ``"Ascend"`` 、``"GPU"`` 和 ``"CPU"``。
        - **device_id** (int) - 目标设备的 ID，值必须在 [0， device_num_per_host-1] 中，默认值： ''0'' 。 ``device_num_per_host`` 指主机上的设备总数。
