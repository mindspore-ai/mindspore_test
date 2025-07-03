mindspore.parallel.sync_pipeline_shared_parameters
============================================================================

.. py:function:: mindspore.parallel.sync_pipeline_shared_parameters(net)

    在流水线并行推理场景下，对stage间的共享权重进行同步。例如 `embedding table` 被 `VocabEmbedding` 和 `LMHead` 两层共享，这两层通常会被切分到不同的stage上。
    在流水线并行推理时， `embedding table` 变更后，有必要在stage之间进行权重同步。

    .. note::
        网络需要先编译，再执行stage之间权重同步。

    参数：
        - **net** (Cell) - 推理网络。

    异常：
        - **TypeError** - `net` 不是 `Cell` 的类型。

    样例：

    .. note::
        运行以下样例之前，需要配置好通信环境变量。

        针对Ascend设备，用户需要编写动态组网启动脚本，详见 `动态组网启动 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dynamic_cluster.html>`_ 。
