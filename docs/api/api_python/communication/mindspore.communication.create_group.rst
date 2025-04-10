mindspore.communication.create_group
====================================

.. py:function:: mindspore.communication.create_group(group, rank_ids)

    创建用户自定义的通信组实例。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - 列表rank_ids的长度应大于1。
        - 列表rank_ids内不能有重复数据。
        - `create_group` 方法应该在 `init` 方法之后使用。
        - 如果没有使用mpirun启动，PyNative模式下仅支持全局单通信组。

    参数：
        - **group** (str) - 输入用户自定义的通信组实例名称，支持数据类型为str。
        - **rank_ids** (list) - 设备编号列表。

    异常：
        - **TypeError** - 参数 `group` 不是字符串或参数 `rank_ids` 不是列表。
        - **ValueError** - 列表rank_ids的长度小于1，或列表 `rank_ids` 内有重复数据，以及后端无效。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
