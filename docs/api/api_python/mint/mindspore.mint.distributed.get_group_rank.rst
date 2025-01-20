mindspore.mint.distributed.get_group_rank
==============================================

.. py:function:: mindspore.mint.distributed.get_group_rank(group, global_rank)

    由通信集群中的全局设备序号获取指定用户通信组中的rank ID。

    .. note::
        `get_group_rank` 方法应该在 :func:`mindspore.mint.distributed.init_process_group` 方法之后使用。

    参数：
        - **group** (str) - 通信组名称，通常由 :func:`mindspore.mint.distributed.new_group` 方法创建，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。
        - **global_rank** (int) - 通信集群内的全局rank ID。

    返回：
        当前通信组内的rank_ID，数据类型为int。

    异常：
        - **TypeError** - 在参数 `global_rank` 不是数字或参数 `group` 不是字符串时抛出。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
