mindspore.hal.memory_reserved
=============================

.. py:function:: mindspore.hal.memory_reserved(device_target=None)

    返回内存池当前管理的内存总量。

    .. note::
        - 接口即将废弃，请使用接口 :func:`mindspore.runtime.memory_reserved` 代替。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。
        - 对于 `CPU` 后端，固定返回0。

    参数：
        - **device_target** (str，可选) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。默认值：``None``。

    返回：
        int，单位为Byte。
