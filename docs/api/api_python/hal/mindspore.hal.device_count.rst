mindspore.hal.device_count
============================

.. py:function:: mindspore.hal.device_count(device_target=None)

    查询指定后端类型的设备数量。

    .. note::
        - 接口即将废弃。
        - CPU环境，请使用接口 :func:`mindspore.device_context.cpu.device_count` 代替。
        - GPU环境，请使用接口 :func:`mindspore.device_context.gpu.device_count` 代替。
        - Ascend环境，请使用接口 :func:`mindspore.device_context.ascend.device_count` 代替。


    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。
        - 对于 ``"CPU"`` 后端，固定返回1。

    参数：
        - **device_target** (str，可选) - 默认值：``None``，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。

    返回：
        int。
