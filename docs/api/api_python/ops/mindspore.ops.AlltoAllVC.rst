mindspore.ops.AlltoAllVC
========================

.. py:class:: mindspore.ops.AlltoAllVC(group=GlobalComm.WORLD_COMM_GROUP)

    相对AllToAll来说，AllToAllVC支持不均匀分散和聚集。相对AllToAllV来说，AllToAllVC性能更好。

    .. note::
        只支持一维的输入，使用该接口前需要将输入数据展开成一维。

    参数：
        - **group** (str) - AlltoAll的通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` ，Ascend平台表示为 ``"hccl_world_group"`` 。
        - **block_size** (int，可选) - 通过 `send_count_matrix` 切分和聚合数据量的基本单位。默认值： ``1`` 。

    输入：
        - **input_x** (Tensor) - 一维待分发的张量, shape为 :math:`(x_1)`。
        - **send_count_matrix** (Union[list[int], Tensor]) - 所有rank的收发参数， :math:`send_count_matrix[i*rank_size+j]` 表示rank i发给rank j的数据量，基本单位是Tensor的数据类型。

    输出：
        Tensor，从每张卡上聚合的一维数据结果。如果结果为空，则返回空张量，且值无意义。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
    
    教程样例：
        - `分布式集合通信原语 - AlltoAllVC
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#alltoallvc>`_