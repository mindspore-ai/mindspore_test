mindspore.parallel.shard
============================================================================

.. py:function:: mindspore.parallel.shard(fn, in_strategy, out_strategy=None, parameter_plan=None)

    指定一个Cell或函数的输入、输出切分策略。在图模式下，可以利用此方法设置某个模块的分布式切分策略，未设置的会自动通过策略传播方式配置。 `in_strategy` 和 `out_strategy` 需要为元组类型，其中的每一个元素指定对应的输入/输出的Tensor分布策略，可参考： :func:`mindspore.ops.Primitive.shard` 的描述。也可以设置为None，会默认以数据并行执行。其余算子的并行策略由输入输出指定的策略推导得到。

    .. note::
        - 仅在半自动并行或自动并行模式下有效。在其他并行模式中，将忽略此处设置的策略。
        - 如果输入含有 `Parameter` ，其对应的策略应该在 `in_strategy` 里设置。

    .. warning::
        该方法当前不支持在PyNative模式下使用。

    参数：
        - **fn** (Union[Cell, Function]) - 待通过分布式并行执行的函数，它的参数和返回值类型应该均为 `Tensor`。 如果 `fn` 是 `Cell` 类型且含有参数，则 `fn` 必须是一个实例化的对象，否则无法访问到其内部参数。
        - **in_strategy** (tuple) - 指定各输入的切分策略，输入元组的每个元素可以为整数元组或 `mindspore.parallel.Layout` 的元组。元组即具体指定输入每一维的切分策略。
        - **out_strategy** (Union[tuple, None]，可选) - 指定各输出的切分策略，用法同 `in_strategy`。默认值： ``None`` 。
        - **parameter_plan** (Union[dict, None]，可选) - 指定各参数的切分策略，传入字典时，格式为 "参数名: 布局"。键是 str 类型的参数名，值是一维整数 `tuple` 或一维 `mindspore.parallel.Layout` 的 `tuple` 表示相应的切分策略。 如果参数名错误或对应参数已经设置了切分策略，该参数的设置会被跳过。该参数只支持对cell类型且含有参数的 `fn` 设置。默认值： ``None`` 。

    返回：
        Function，返回一个在自动并行流程下执行的函数。

    异常：
        - **AssertionError** - 如果并行模式不是"auto_parallel"或"semi_auto_parallel"。
        - **TypeError** - 如果 `in_strategy` 不是tuple。
        - **TypeError** - 如果 `out_strategy` 不是tuple或None。
        - **TypeError** - 如果 `in_strategy` 里的任何一个元素不是tuple(int)或者tuple(mindspore.parallel.Layout)。
        - **TypeError** - 如果 `out_strategy` 里的任何一个元素不是tuple(int)或者tuple(mindspore.parallel.Layout)。
        - **TypeError** - 如果 `parameter_plan` 不是dict或None。
        - **TypeError** - 如果 `parameter_plan` 里的任何一个键值类型不是str。
        - **TypeError** - 如果 `parameter_plan` 里的任何一个值类型不是tuple(int)或者tuple(mindspore.parallel.Layout)。
