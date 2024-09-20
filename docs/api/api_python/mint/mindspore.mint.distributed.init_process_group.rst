mindspore.mint.distributed.init_process_group
=================================================

.. py:function:: mindspore.mint.distributed.init_process_group(backend="hccl",init_method=None,timeout=None,world_size=-1,rank=-1,store=None,pg_option=None,device_id=None)

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
        - **TypeError** - 参数 `backend` 的值不是hccl。
        - **TypeError** - 参数 `world_size` 的值非-1，且与实际通讯集群数不一致。
        - **RuntimeError** - 1）硬件设备类型无效；2）后台服务无效；3）分布式计算初始化失败；4）后端是HCCL的情况下，未设置环境变量 RANK_ID 或 MINDSPORE_HCCL_CONFIG_PATH 的情况下初始化HCCL服务。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
