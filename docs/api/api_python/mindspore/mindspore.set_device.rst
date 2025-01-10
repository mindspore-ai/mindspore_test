mindspore.set_device
=====================

.. py:function:: mindspore.set_device(device_target, device_id=None)

    设置运行环境的设备类型和设备ID。

    .. note::
        - `device_target` 的取值必须在 ["CPU", "GPU", "Ascend"] ，没有默认值。
    
    参数：
        - **device_target** (str) - 要运行的目标设备，仅支持 ``"Ascend"`` 、``"GPU"`` 和 ``"CPU"``。
        - **device_id** (int) - 目标设备的 ID，值必须在 [0， device_num_per_host-1] 中。默认值: ``None``，框架将根据场景按需设置不同的默认行为：如果是单卡场景，则框架设置为0；如果是msrun启动的分布式场景下，则框架会自动协商可用的device_id值；如果是其他启动方式的分布式场景下，则框架设置为0。 ``device_num_per_host`` 指主机上的设备总数。