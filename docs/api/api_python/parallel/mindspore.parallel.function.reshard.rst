mindspore.parallel.function.reshard
============================================================================

.. py:function:: mindspore.parallel.function.reshard(tensor, layout)

    将张量从一种分布式排布转换成另一种分布式排布。其中，传入的layout需要为mindspore.parallel.Layout类型，可参考： :class:`mindspore.parallel.Layout` 的描述。

    .. note::
        在图模式下，可以利用此方法设置某个张量的并行切分策略，未设置的会自动通过策略传播方式配置。

    .. warning::
        该方法当前不支持在PyNative模式下使用。

    参数：
        - **tensor** (Tensor) - 待设置切分策略的张量。
        - **layout** (Layout) - 指定精准排布的方案，包括描述设备的排布（device_matrix）和设备矩阵的映射别名（alias_name）。

    返回：
        Tensor，与输入的tensor数学等价。

    异常：
        - **TypeError** - 输入参数 `tensor` 不是mindspore.Tensor类型。
        - **TypeError** - 输入参数 `layout` 不是mindspore.parallel.Layout类型。

    样例：

    .. note::
        运行以下样例之前，需要配置好通信环境变量。

        针对Ascend/GPU/CPU设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html>`_ 。

        该样例需要在8卡环境下运行。