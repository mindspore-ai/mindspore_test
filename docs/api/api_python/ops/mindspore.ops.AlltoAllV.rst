mindspore.ops.AlltoAllV
=======================

.. py:class:: mindspore.ops.AlltoAllV(group=GlobalComm.WORLD_COMM_GROUP)

    相对AlltoAll来说，AlltoAllV算子支持不等分的切分和聚合。

    .. note::
        只支持一维的输入，使用该接口前需要将输入数据展开成一维。

    参数：
        - **group** (str) - AlltoAll的通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` ，Ascend平台表示为 ``"hccl_world_group"`` 。

    输入：
        - **input_x** (Tensor) - 一维待分发的张量, shape为 :math:`(x_1)`。
        - **send_numel_list** (Union[tuple[int], list[int], Tensor]) - 分发给每张卡的数据量。
        - **recv_numel_list** (Union[tuple[int], list[int], Tensor]) - 从每张卡聚合的数据量。

    输出：
        Tensor，从每张卡上聚合的一维数据结果。如果结果为空，则范围无意义的数值0。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
    
    教程样例：
        - `分布式集合通信原语 - AlltoAllV
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#alltoallv>`_