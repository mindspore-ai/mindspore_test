mindspore.ops.ForiLoop
======================

.. py:class:: mindspore.ops.ForiLoop

    一段范围内的循环操作

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **lower** (Union[int, Tensor]) - 循环的起始索引值
        - **upper** (Union[int, Tensor]) - 循环的结束索引值
        - **loop_func** (Function) - 循环体函数
        - **init_val** (Union[Tensor, Number, Str, Bool, List, Tuple, Dict]) - 循环的初始值
        - **unroll** (Optional) Bool - 是否在编译阶段展开

    输出：
        Union[Tensor, Number, Str, Bool, List, Tuple, Dict] 循环的最终结果，和`init_val`的类型相同

    异常：
        - **TypeError** - `lower` 不是一个整数或者Tensor
        - **TypeError** - `upper` 不是一个整数或者Tensor
        - **TypeError** - `loop_func` 不是一个函数。
        - **ValueError** - `loop_func` 不能接受`init_val`作为参数或者返回值和`init_val`的类型不同
