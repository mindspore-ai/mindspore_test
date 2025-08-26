mindspore.parallel.strategy.get_strategy_metadata
============================================================================

.. py:function:: mindspore.parallel.strategy.get_strategy_metadata(network, rank_id=None)

    获取当前网络的所有卡的在线策略信息。
    关于 `Layout` 的解释，请参考 :class:`mindspore.parallel.Layout`。

    参数：
        - **network** (str) - 训练网络的名称。
        - **rank_id** (int, 可选) - 指定卡号。默认为 ``None``，表示返回所有卡的策略。

    返回：
        Dict，返回一个字典，包含所有卡或特定卡的参数切分策略信息。
        字典的键是 `rank_id`，值是该卡所有参数的切分策略。
        在每个卡的策略中，key 是参数名称，value 是该参数的切分策略。
        如果指定了 `rank_id`，字典将返回该卡的策略信息；否则，返回网络中所有卡的策略信息。
        不支持场景，则返回 ``None``。
