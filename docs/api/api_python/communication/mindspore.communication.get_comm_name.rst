mindspore.communication.get_comm_name
=====================================

.. py:function:: mindspore.communication.get_comm_name(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通讯组的通讯器名称。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - `get_comm_name` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str) - 传入的通信组名称，通常由 `create_group` 方法创建，或默认使用 ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        str，指定通讯组的通讯器名称。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **ValueError** - 在后端不可用时抛出。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

