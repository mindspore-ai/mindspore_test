mindspore.device_context.ascend.op_tuning.aoe_tune_mode
=======================================================

.. py:function:: mindspore.device_context.ascend.op_tuning.aoe_tune_mode(tune_mode)

    表示启动AOE调优，默认不设置。

    参数：
        - **tune_mode** (str) - AOE调优模式配置。

          - ``"online"``: 启动在线调优。
          - ``"offline"``: 为离线调优保存GE图。