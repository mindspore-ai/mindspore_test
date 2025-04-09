mindspore.communication.destroy_group
=====================================

.. py:function:: mindspore.communication.destroy_group(group)

    注销用户通信组。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - 参数 `group` 不能是 ``"hccl_world_group"``。
        - `destroy_group` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str) - 被注销通信组实例（通常由 create_group 方法创建）的名称。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **ValueError** - 在参数 `group` 是 ``"hccl_world_group"`` 或后端不可用时抛出。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst