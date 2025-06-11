mindspore.nn.PipelineCell
=========================

.. py:class:: mindspore.nn.PipelineCell(network, micro_size, stage_config=None, segment_config=None)

    将MiniBatch切分成更细粒度的MicroBatch，用于流水线并行的训练中。

    .. note::
        - micro_size必须大于或等于流水线stage的个数。
        - 接口即将废弃，请使用接口 :class:`mindspore.parallel.nn.Pipeline` 代替。

    参数：
        - **network** (Cell) - 要修饰的目标网络。
        - **micro_size** (int) - MicroBatch大小。
        - **stage_config** (dict，可选) - 流水线并行对于每个cell的stage配置。默认值： ``None``。
        - **segment_config** (dict，可选) - 流水线并行对于每个cell的segment配置。默认值： ``None``。
