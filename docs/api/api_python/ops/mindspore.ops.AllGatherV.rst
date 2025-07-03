mindspore.ops.AllGatherV
=========================

.. py:class:: mindspore.ops.AllGatherV(group=GlobalComm.WORLD_COMM_GROUP)

    从指定的通信组中收集不均匀的张量，并返回全部收集的张量。

    .. note::
        只支持一维的输入，使用该接口前需要将输入数据展开成一维。

    参数：
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    输入：
        - **input_x** (Tensor) - 一维待汇聚的张量，shape为 :math:`(x_1)`。
        - **output_split_sizes** (Union[tuple[int], list[int], Tensor]) - 一维张量，所有rank的汇聚数据量列表，基本单位是Tensor的数据类型。

    输出：
        Tensor，从每张卡上聚合的一维数据结果。如果结果为空，则返回空张量，且值无意义。

    异常：
        - **RuntimeError** - 目标设备无效、后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。

    教程样例：
        - `分布式集合通信原语 - AllGatherV
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#allgatherv>`_