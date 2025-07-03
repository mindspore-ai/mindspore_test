mindspore.parallel.nn.PipelineGradReducer
============================================================================

.. py:class:: mindspore.parallel.nn.PipelineGradReducer(parameters, scale_sense=1.0, opt_shard=None)

    函数式训练场景下，实现流水线并行的梯度规约及累加。

    参数：
        - **parameters** (list) - 将进行pp并行的网络参数。
        - **scale_sense** (float，可选) - 梯度的尺度感知。默认值： ``1.0``。
        - **opt_shard** (bool，可选) - 如果使用优化器，需要配置为True。默认值： ``None``。

    异常：
        - **RuntimeError** - `mode` 不是图模式。

    样例：

    .. note::
        在运行以下示例之前，您需要配置通信环境变量。对于 Ascend 设备，用户需要准备 rank 表，并设置 rank_id 和 device_id。请参阅 `rank table 启动 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/rank_table.html>`_ 以获取更多详细信息。此示例需要在多设备上运行。
