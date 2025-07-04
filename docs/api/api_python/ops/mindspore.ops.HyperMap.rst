mindspore.ops.HyperMap
=======================

.. py:class:: mindspore.ops.HyperMap(ops=None, reverse=False)

    对输入序列做集合运算。

    对序列的每个元素或嵌套序列进行运算。与 :class:`mindspore.ops.Map` 不同，`HyperMap` 能够用于嵌套结构。

    参数：
        - **ops** (Union[MultitypeFuncGraph, None]，可选) -  `ops` 是指定运算操作。如果 `ops` 为 ``None`` ，则运算应该作为 `HyperMap` 实例的第一个入参。默认值为 ``None`` 。
        - **reverse** (bool，可选) -  在某些场景下，需要逆向以提高计算的并行性能，一般情况下，用户可以忽略。 `reverse` 用于决定是否逆向执行运算，仅在图模式下支持。默认值为 ``False`` 。

    输入：
        - **args** (Tuple[sequence]) -

          - 如果 `ops` 不是 ``None`` ，则所有入参都应该是具有相同长度的序列，并且序列的每一行都是运算的输入。
          - 如果 `ops` 是 ``None`` ，则第一个入参是运算，其余都是输入。

    .. note::
        输入数量等于 `ops` 的输入数量。

    输出：
        序列或嵌套序列，执行 `operation(args[0][i], args[1][i])` 之后输出的序列，其中 `operation` 为 `ops` 指定的一个函数。

    异常：
        - **TypeError** - 如果 `ops` 既不是 :class:`mindspore.ops.MultitypeFuncGraph` 也不是 ``None`` 。
        - **TypeError** - 如果 `args` 不是一个tuple。
