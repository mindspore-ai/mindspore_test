mindspore.mint.distributed.get_world_size
============================================

.. py:function:: mindspore.mint.distributed.get_world_size(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组实例的rank_size。

    .. note::
        - `get_world_size` 方法应该在 `init_process_group` 方法之后使用。

    参数：
        - **group** (str) - 指定工作组实例（由 mindspore.communication.create_group 方法创建）的名称，支持数据类型为str，默认值为 ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        指定通信组实例的rank_size，数据类型为int。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **ValueError** - 在后台不可用时抛出。
        - **RuntimeError** - 在 `HCCL` 服务不可用时抛出。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst