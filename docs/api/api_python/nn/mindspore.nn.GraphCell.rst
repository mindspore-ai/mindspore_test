mindspore.nn.GraphCell
======================

.. py:class:: mindspore.nn.GraphCell(graph, params_init=None, obf_random_seed=None)

    运行从MindIR加载的计算图。

    此功能仍在开发中。目前 `GraphCell` 不支持修改图结构，在导出MindIR时只能使用shape和类型与输入相同的数据。

    参数：
        - **graph** (FuncGraph) - 从MindIR加载的编译图。
        - **params_init** (dict) - 需要在图中初始化的参数。key为参数名称，类型为字符串，value为 Tensor 或 Parameter。如果参数名在图中已经存在，则更新其值；如果不存在，则忽略。默认值： ``None`` 。
        - **obf_random_seed** (Union[int, None]) - 用于动态混淆保护的混淆随机种子，目前动态结构混淆已不支持。

    异常：
        - **NotImplementedError** - 动态结构混淆已不支持。
        - **TypeError** - 如果图不是FuncGraph类型。
        - **TypeError** - 如果 `params_init` 不是字典。
        - **TypeError** - 如果 `params_init` 的key不是字符串。
        - **TypeError** - 如果 `params_init` 的value既不是 Tensor也不是Parameter。
