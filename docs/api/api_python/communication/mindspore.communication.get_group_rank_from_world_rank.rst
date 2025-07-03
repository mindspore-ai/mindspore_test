mindspore.communication.get_group_rank_from_world_rank
======================================================

.. py:function:: mindspore.communication.get_group_rank_from_world_rank(world_rank_id, group)

    由通信集群中的全局设备序号获取指定用户通信组中的rank ID。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - 参数 `group` 不能是 ``"hccl_world_group"``。
        - `get_group_rank_from_world_rank` 方法应该在 `init` 方法之后使用。

    参数：
        - **world_rank_id** (`int`) - 通信集群内的全局rank ID。
        - **group** (`str`) - 指定通信组实例（由 `create_group` 方法创建）的名称。

    返回：
        当前通信组内的rank_ID，数据类型为int。

    异常：
        - **TypeError** - 在参数 `world_rank_id` 不是数字或参数 `group` 不是字符串时抛出。
        - **ValueError** - 在参数 `group` 是 ``"hccl_world_group"`` 或后端不可用时抛出。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst