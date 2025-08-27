mindspore.parallel.strategy.get_current_strategy_metadata
============================================================================

.. py:function:: mindspore.parallel.strategy.get_current_strategy_metadata(network)

    获取当前网络的当前卡的在线策略信息。

    参数：
        - **network** (str) - 训练网络的名称。

    返回：
        Dict，key为0，value为当前卡的所有参数的切分策略。value中的key为参数名称，value为对应参数的切分策略信息。
        不支持场景，则返回 ``None``。