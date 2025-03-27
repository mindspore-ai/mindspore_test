mindspore.parallel.sync_pipeline_shared_parameters
============================================================================

.. py:function:: mindspore.parallel.sync_pipeline_shared_parameters(net)

    推理场景下，实现不同stage之间共享权重。

    .. note::
        在同步流水线并行阶段共享参数之前，应编译网络。

    参数：
        - **net** (Cell) - 将进行共享权重的网络。

    异常：
        - **TypeError** - 输入的 `net` 不是Cell模式。
