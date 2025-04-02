mindspore.parallel.set_op_strategy_config
=========================================

.. py:function:: mindspore.parallel.set_op_strategy_config(mode="SAVE", path="")

    自动并行策略传播模式下，通过此接口设置保存或加载算子级策略。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除，推荐使用新接口 :func:`mindspore.parallel.auto_parallel.AutoParallel.load_operator_strategy_file` 和 :func:`mindspore.parallel.auto_parallel.AutoParallel.save_operator_strategy_file`；
        - 该接口暂不支持保存或加载Layout形式的算子策略。
    
    .. note::
        - 仅在自动并行模式且搜索模式为策略传播时有效。
        - 仅支持相同网络相同配置下保存再加载，使用保存模式保存算子策略至json文件后，若修改了网络结构或训练超参数，可能导致使用加载功能失败。
        - 进行分布式训练时，用户可以先用单机dryrun功能保存策略，再使用加载功能进行分布式训练。

    参数：
        - **mode** (str) - 描述模式为保存还是加载， ``"SAVE"`` 时保存算子策略至设置的json文件， ``"LOAD"`` 时从json文件中加载算子策略。默认值： ``"SAVE"`` 。
        - **path** (str) - 描述策略保存或者加载的json文件路径，只支持绝对路径。 默认值： ``""`` 。

    异常：
        - **KeyError** - `mode` 不是 ``"SAVE"`` 或 ``"LOAD"`` 。
        - **KeyError** - `path` 不是以 ``".json"`` 结尾。
        - **KeyError** - `path` 不是绝对路径。
