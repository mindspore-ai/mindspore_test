mindspore.communication
========================
集合通信接口。

注意，集合通信接口需要先配置好通信环境变量。

针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。

.. py:function:: mindspore.distributed.init_process_group(backend="hccl",init_method=None,timeout=None,world_size=-1,rank=-1,store=None,pg_option=None,device_id=None)

    初始化通信服务并创建默认通讯group（group=GlobalComm.WORLD_COMM_GROUP）。

    .. note::
        - 当前接口不支持GPU、CPU版本的Mindpore调用。

    参数：
        - **backend** (str，可选参数) - 分布式后端的名称，默认为 ``"hccl"``，且目前和只能设置为hccl。
        - **init_method** (str, 无效参数) - 初始化通讯域时的URL配置。这个参数主要对标PT，但实际无效且设置不生效
        - **timeout** (int, 无效参数) - 设置接口的超时配置。这个参数主要对标PT，但实际无效且设置不生效
        - **world_size** (int, 可选参数) - 配置初始化时全局通讯的卡数
        - **rank** (int, 无效参数) - 设置当前设备卡的卡号。这个参数主要对标PT，但实际无效且设置不生效
        - **store** (store, 无效参数) - key/value 数据在设备进程上存储以便于设备进程间通讯地址、连接信息的交换。这个参数主要对标PT，但实际无效且设置不生效
        - **pg_option** (ProcessGroupOption, 无效参数) - 针对创建的通讯group设置特殊配置策略。这个参数主要对标PT，但实际无效且设置不生效
        - **device_id** (int, 无效参数) - 设置当前进程使用的NPU卡卡号。这个参数主要对标PT，但实际无效且设置不生效

    异常：
        - **TypeError** - 参数 `backend`的值不是hccl。
        - **TypeError** - 参数 `world_size`的值非-1，且与世纪通讯集群数不一致。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

.. py:function:: mindspore.distributed.destroy_process_group(group=None)
    
    销毁指定通讯group
    如果指定的通讯group为None或“hccl_world_group”, 则销毁全局通讯域并释放分布式资源，例如 `HCCL` 或 `NCCL` 或 `MCCL` 服务。

    .. note::
        - 此方法应该在 `init_process_group` 方法之后使用。

    参数：
        - **group** (str) - 被注销通信组实例（通常由 mindspore.communication.create_group 方法创建）的名称。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

.. py:function:: mindspore.distributed.get_rank(group=GlobalComm.WORLD_COMM_GROUP)

    在指定通信组中获取当前的设备序号。

    .. note::
        - `get_rank` 方法应该在 `init_process_group` 方法之后使用。

    参数：
        - **group** (str) - 通信组名称，通常由 `mindspore.communication.create_group` 方法创建，否则将使用默认组。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        int，调用该方法的进程对应的组内序号。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **ValueError** - 在后台不可用时抛出。
        - **RuntimeError** - 在 `HCCL` 服务不可用时抛出。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

.. py:function:: mindspore.distributed.get_world_size(group=GlobalComm.WORLD_COMM_GROUP)

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
        .. include:: ops/mindspore.ops.comm_note.rst

