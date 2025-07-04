mindspore.ops.AllGather
========================

.. py:class:: mindspore.ops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)

    在指定的通信组中汇聚Tensor，返回汇聚后的张量。

    .. note::
        - 集合中所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    输入：
        - **input_x** (Tensor) - AllGather的输入，shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。

    输出：
        Tensor，如果组中的device数量为N，则输出的shape为 :math:`(N*x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - `group` 不是str。
        - **ValueError** - 调用进程的rank id大于本通信组的rank大小。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。

    教程样例：
        - `分布式集合通信原语 - AllGather
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#allgather>`_
