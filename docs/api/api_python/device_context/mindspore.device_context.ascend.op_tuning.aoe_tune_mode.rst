mindspore.device_context.ascend.op_tuning.aoe_tune_mode
=======================================================

.. py:function:: mindspore.device_context.ascend.op_tuning.aoe_tune_mode(tune_mode)

    表示启动AOE调优，默认不设置。
    如果您想了解更多详细信息，
    请查询 `AOE调优工具 <https://www.mindspore.cn/docs/zh-CN/master/model_train/optimize/aoe.html>`_ 了解。

    参数：
        - **tune_mode** (str) - AOE调优模式配置。

          - ``"online"``: 启动在线调优。
          - ``"offline"``: 为离线调优保存GE图。

    支持平台：
        ``Ascend``