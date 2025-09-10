mindspore.train.AdaptiveCheckpointConfig
=========================================

.. py:class:: mindspore.train.AdaptiveCheckpointConfig(target_overhead_percentage=None, failure_rate=None, **kwargs)

    自适应checkpoint保存的配置策略。

    该类扩展了CheckpointConfig，支持基于目标开销百分比或故障率的自适应checkpoint保存。

    .. note::
        只能设置 `target_overhead_percentage` 或 `failure_rate` 中的一个，不能同时设置。

    参数：
        - **target_overhead_percentage** (float, 可选) - checkpoint保存应消耗的训练时间的目标百分比。设置后，checkpoint保存间隔将动态调整以维持此开销百分比。默认值： ``None`` 。
        - **failure_rate** (float, 可选) - 用于最优checkpoint间隔计算的预期故障率。设置后，使用公式：interval = sqrt(2 * checkpoint_time / (failure_rate * step_time)) 来确定最优保存频率。默认值： ``None`` 。
        - **kwargs** - 传递给CheckpointConfig的其他参数。

    异常：
        - **ValueError** - 如果同时设置了 `target_overhead_percentage` 和 `failure_rate` 。
        - **ValueError** - 如果 `target_overhead_percentage` 不在有效范围 (0, 100] 内。
        - **ValueError** - 如果 `failure_rate` 不在有效范围 (0, 1] 内。

    .. py:method:: is_adaptive
        :property:

        检查是否启用了自适应checkpoint保存。

        返回：
            bool，是否启用了自适应checkpoint保存。

    .. py:method:: adaptive_mode
        :property:

        获取自适应模式。

        返回：
            str，自适应模式，可以是 ``'percentage'`` 或 ``'failure_rate'``，如果未启用自适应则返回 ``None``。