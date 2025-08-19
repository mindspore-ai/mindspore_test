mindspore.recompute
===================

.. py:function:: mindspore.recompute(block, *args, **kwargs)

    该函数用于减少显存的使用，当运行选定的模块时，不再保存其中的前向计算产生的激活值，我们将在反向传播时，重新计算前向的激活值。

    .. note::
        重计算函数只支持继承自Cell对象的模块。

    参数：
        - **block** (Cell) - 需要重计算的网络模块。
        - **args** (tuple) - 指需要重计算的网络模块的前向输入。
        - **kwargs** (dict) - 可选输入。

    返回：
        同block的返回类型相同。

    异常：
        - **TypeError** - 如果 `block` 不是Cell对象。

