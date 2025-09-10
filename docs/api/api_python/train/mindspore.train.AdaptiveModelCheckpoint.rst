mindspore.train.AdaptiveModelCheckpoint
========================================

.. py:class:: mindspore.train.AdaptiveModelCheckpoint(prefix='CKP', directory=None, config=None)

    自适应checkpoint的回调函数。

    该类扩展了ModelCheckpoint，支持基于训练性能指标动态调整保存间隔的自适应checkpoint保存。

    参数：
        - **prefix** (Union[str, callable object], 可选) - checkpoint文件的前缀名称，或者用来生成名称的可调用对象。默认值： ``'CKP'`` 。
        - **directory** (Union[str, callable object], 可选) - 保存checkpoint文件的文件夹路径，或者用来生成路径的可调用对象。默认情况下，文件保存在当前目录下。默认值： ``None`` 。
        - **config** (AdaptiveCheckpointConfig, 可选) - 自适应checkpoint策略配置。默认值： ``None`` 。

    异常：
        - **TypeError** - 如果config不是AdaptiveCheckpointConfig类型。

    .. py:method:: step_begin(run_context)

        在step开始时记录时间，用于自适应计时。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: step_end(run_context)

        在step结束时保存checkpoint并更新自适应间隔。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。