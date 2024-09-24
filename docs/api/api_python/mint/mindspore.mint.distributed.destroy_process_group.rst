mindspore.mint.distributed.destroy_process_group
==================================================

.. py:function:: mindspore.mint.distributed.destroy_process_group(group=None)

    销毁指定通讯group。
    如果指定的通讯group为None或“hccl_world_group”, 则销毁全局通讯域并释放分布式资源，例如 `hccl`  服务。

    .. note::
        - 此方法应该在 `init_process_group` 方法之后使用。

    参数：
        - **group** (str) - 被注销通信组实例（通常由 mindspore.communication.create_group 方法创建）的名称。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst