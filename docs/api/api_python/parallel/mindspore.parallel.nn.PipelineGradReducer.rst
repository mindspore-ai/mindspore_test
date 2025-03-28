mindspore.parallel.nn.PipelineGradReducer
============================================================================

.. py:class:: mindspore.parallel.nn.PipelineGradReducer(optimizer.parameters, scale_sense=1.0, opt_shard=True)

    流水线并行（pp）梯度累加的GradReducer。

    参数：
        - **parameters** (list) - 将进行pp并行的网络。
        - **scale_sense** (float，可选) - 梯度的尺度感知。默认值： ``1.0``。
        - **opt_shard** (bool，可选) - 如果使用优化器，需要配置为True。默认值： ``True``。

    异常：
        - **RuntimeError** - `mode` 不是图模式。
