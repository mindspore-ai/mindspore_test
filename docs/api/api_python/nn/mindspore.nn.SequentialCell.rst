mindspore.nn.SequentialCell
============================

.. py:class:: mindspore.nn.SequentialCell(*args)

    构造Cell顺序容器。关于Cell的介绍，可参考 `Cell <https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>`_。

    SequentialCell将按照传入List的顺序依次添加Cell。此外，也支持OrderedDict作为构造器传入。

    参数：
        - **args** (list, OrderedDict) - 仅包含Cell子类的列表或有序字典。

    输入：
        - **x** (Tensor) - Tensor，其shape取决于序列中的第一个Cell。

    输出：
        Tensor，输出Tensor，其shape取决于输入 `x` 和定义的Cell序列。

    异常：
        - **TypeError** - `args` 的类型不是列表或有序字典。

    .. py:method:: append(cell)

        在容器末尾添加一个Cell。

        参数：
            - **cell** (Cell) - 要添加的Cell。
