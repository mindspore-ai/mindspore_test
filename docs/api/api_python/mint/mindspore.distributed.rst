mindspore.communication
========================
集合通信接口。

注意，集合通信接口需要先配置好通信环境变量。

针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。

.. py:function:: mindspore.distributed.init_process_group(backend="hccl",init_method=None,timeout=None,world_size=-1,rank=-1,store=None,pg_option=None,device_id=None)

    初始化通信服务并创建默认通讯group（group=GlobalComm.WORLD_COMM_GROUP）。

    .. note::
        - 当前接口不支持GPU、CPU版本的Mindpore调用。
        - 在Ascend硬件平台下，这个接口的设置需要在创建Tensor和Parameter之前，以及所有算子和网络的实例化和运行之前。

    参数：
        - **backend** (str，可选参数) - 分布式后端`的名称，默认为 ``"hccl"``，且目前只能设置为hccl。
        - **init_method** (str, 无效参数) - 初始化通讯域时的URL配置。这个参数主要对标PT，但实际无效且设置不生效。
        - **timeout** (int, 无效参数) - 设置接口的超时配置。这个参数主要对标PT，但实际无效且设置不生效。
        - **world_size** (int, 可选参数) - 配置初始化时全局通讯的卡数。
        - **rank** (int, 无效参数) - 设置当前设备卡的卡号。这个参数主要对标PT，但实际无效且设置不生效。
        - **store** (store, 无效参数) - key/value 数据在设备进程上存储以便于设备进程间通讯地址、连接信息的交换。这个参数主要对标PT，但实际无效且设置不生效。
        - **pg_option** (ProcessGroupOption, 无效参数) - 针对创建的通讯group设置特殊配置策略。这个参数主要对标PT，但实际无效且设置不生效。
        - **device_id** (int, 无效参数) - 设置当前进程使用的NPU卡卡号。这个参数主要对标PT，但实际无效且设置不生效。

    异常：
        - **TypeError** - 参数 `backend`的值不是hccl。
        - **TypeError** - 参数 `world_size`的值非-1，且与实际通讯集群数不一致。
        - **RuntimeError** - 1）硬件设备类型无效；2）后台服务无效；3）分布式计算初始化失败；4）后端是HCCL的情况下，未设置环境变量 RANK_ID 或 MINDSPORE_HCCL_CONFIG_PATH 的情况下初始化HCCL服务。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

.. py:function:: mindspore.distributed.destroy_process_group(group=None)
    
    销毁指定通讯group。
    如果指定的通讯group为None或“hccl_world_group”, 则销毁全局通讯域并释放分布式资源，例如 `hccl`  服务。

    .. note::
        - 此方法应该在 `init_process_group` 方法之后使用。

    参数：
        - **group** (str) - 被注销通信组实例（通常由 mindspore.communication.create_group 方法创建）的名称。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

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
        .. include:: ../ops/mindspore.ops.comm_note.rst

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
        .. include:: ../ops/mindspore.ops.comm_note.rst

.. py:function:: mindspore.mint.distributed.P2POp(op, tensor, peer, group=None, tag=0, *, recv_dtype=None)

    用于存放关于'isend'、'irecv'相关的信息， 并用于 `batch_isend_irecv` 接口的入参。

    .. note::
        - 当 `op` 入参为'irecv'时， `tensor` 入参允许不传入张量类型， 可以只传入接收张量的形状。
        - `tensor` 入参不会被最后的结果原地修改。

    参数：
        - **op** (Union[str, function]) - 对于字符串类型，只允许'isend'和'irecv'。 对于函数类型，只允许 ``comm_func.isend`` 和 ``comm_func.irecv`` 函数。
        - **tensor** (Union[Tensor, Tuple(int)]) - 用于发送或接收的张量。 如果是 `op` 是'irecv'，可以传入接收张量的形状。
        - **peer** (int) - 发送或接收的远程设备的全局编号。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。
        - **tag** (int，可选) - 当前暂不支持。 默认值：0。

    关键字参数：
        - **recv_dtype** (mindspore.dtype，可选) - 表示接收张量的数据类型。 当 `tensor` 传入的是张量的形状时，该入参必须要配置。默认值：``None``。

    返回：
        `P2POp` 对象。

    异常：
        - **ValueError** - 当 `op` 不是与'isend'和'irecv'相关的字符串或函数。
        - **TypeError** - 当 `tensor` 不是张量或者元组类型。
        - **NotImplementedError** - 当 `tag` 入参不为0。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。

.. py:function:: mindspore.mint.distributed.batch_isend_irecv(p2p_op_list)

    异步地发送和接收张量。

    .. note::
        - 不同设备中， `p2p_op_list` 中的 `P2POp` 的 ``"isend`` 和 ``"irecv"`` 应该互相匹配。
        - `p2p_op_list` 中的 `P2POp` 应该使用同一个通信组。
        - 暂不支持 `p2p_op_list` 中的 `P2POp` 含有 `tag` 入参。
        - `p2p_op_list` 中的 `P2POp` 的 `tensor` 的值不会被最后的结果原地修改。
        - 仅支持PyNative模式，目前不支持Graph模式。

    参数：
        - **p2p_op_list** (P2POp) - 包含 `P2POp` 类型对象的列表。 `P2POp` 指的是 :class:`mindspore.mint.distributed.P2POp`。

    返回：
        Tuple(Tensor)。根据 `p2p_op_list` 中的 `P2POp` 的发送/接收顺序，得到的接收张量元组。
        当 `P2POp` 为发送时， 相应位置的结果是没有意义的张量。
        当 `P2POp` 为接收时， 相应位置的结果是从其他设备接收到的张量。

    异常：
        - **TypeError** - `p2p_op_list` 中不全是 `P2POp` 类型。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
