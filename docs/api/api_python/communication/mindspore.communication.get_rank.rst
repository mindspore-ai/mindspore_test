mindspore.communication.get_rank
================================

.. py:function:: mindspore.communication.get_rank(group=GlobalComm.WORLD_COMM_GROUP)

    在指定通信组中获取当前的设备序号。

    .. note::
        - `get_rank` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str) - 通信组名称，通常由 `create_group` 方法创建，否则将使用默认组。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        int，调用该方法的进程对应的组内序号。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **ValueError** - 在后端不可用时抛出。
        - **RuntimeError** - 在 `HCCL` 或 `NCCL` 或 `MCCL` 服务不可用时抛出。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
