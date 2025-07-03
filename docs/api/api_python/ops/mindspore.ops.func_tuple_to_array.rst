mindspore.ops.tuple_to_array
==============================

.. py:function:: mindspore.ops.tuple_to_array(input_x)

    将tuple转换为tensor。

    .. note::
        如果tuple中第一个数据类型为int，则输出tensor的数据类型为int。否则，输出tensor的数据类型为float。

    参数：
        - **input_x** (tuple) - 数值型组成的tuple。仅支持常量值。

    返回：
        Tensor
