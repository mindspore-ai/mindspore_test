mindspore.hal.get_arch_list
=============================

.. py:function:: mindspore.hal.get_arch_list(device_target=None)

    返回此MindSpore包支持哪些后端架构，此接口将在后续版本中废弃。

    参数：
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        GPU设备返回str，Ascend以及CPU设备返回None。
