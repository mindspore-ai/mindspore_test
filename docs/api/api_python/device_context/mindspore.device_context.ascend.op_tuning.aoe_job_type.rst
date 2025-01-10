mindspore.device_context.ascend.op_tuning.aoe_job_type
======================================================

.. py:function:: mindspore.device_context.ascend.op_tuning.aoe_job_type(config)

    设置AOE工具专用的参数，需要和mindspore.device_context.op_tuning.aoe_tune_mode(tune_mode)配合使用，框架默认设置为2。
    如果您想了解更多详细信息，
    请查询 `AOE调优工具 <https://www.mindspore.cn/docs/zh-CN/master/model_train/optimize/aoe.html>`_ 了解。

    参数：
        - **config** (str) - 选择调优类型。

          - ``"1"``: 设置为子图调优。
          - ``"2"``: 设置为算子调优。