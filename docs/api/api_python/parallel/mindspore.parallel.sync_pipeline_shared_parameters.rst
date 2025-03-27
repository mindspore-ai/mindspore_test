mindspore.parallel.sync_pipeline_shared_parameters
============================================================================

.. py:function:: mindspore.parallel.sync_pipeline_shared_parameters(net)

    推理场景下，实现不同stage之间共享权重。

    参数：
        - **net** (Cell) - 将进行共享权重的网络。

    异常：
        - **TypeError** - 输入的 `net` 不是Cell模式。

    样例：

    .. note::
        运行以下样例之前，需要配置好通信环境变量。

        针对Ascend设备，用户需要编写动态组网的启动脚本，详见 `Dynamic Cluster
        Startup <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dynamic_cluster.html>`_ 。
