mindspore.communication.get_group_size
======================================

.. py:function:: mindspore.communication.get_group_size(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组实例的rank_size。

    .. note::
        - `get_group_size` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str) - 指定工作组实例（由 create_group 方法创建）的名称，支持数据类型为str，默认值为 ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        指定通信组实例的rank_size，数据类型为int。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **ValueError** - 在后端不可用时抛出。
        - **RuntimeError** - 在 `HCCL` 或 `NCCL` 或 `MCCL` 服务不可用时抛出。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
