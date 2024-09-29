mindspore.mint.distributed.get_rank
=====================================

.. py:function:: mindspore.mint.distributed.get_rank(group=None)

    在指定通信组中获取当前的设备序号。

    .. note::
        - `get_rank` 方法应该在 `init_process_group` 方法之后使用。

    参数：
        - **group** (str) - 通信组名称，通常由 `mindspore.communication.create_group` 方法创建，否则将使用默认组。如果设置为 ``None`` ，会使用 ``GlobalComm.WORLD_COMM_GROUP`` 。默认值： ``None`` 。

    返回：
        int，调用该方法的进程对应的组内序号。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst