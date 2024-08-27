mindspore.ops.WhileLoop
=======================

.. py:class:: mindspore.ops.WhileLoop

    在编译阶段不进行循环展开的循环算子。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **cond_func** (Function) - 循环的条件函数。
        - **loop_func** (Function) - 循环体函数，接受一个参数，并且返回值与输入参数的类型相同。
        - **init_val** (Union[Tensor, number, str, bool, list, tuple, dict]) - 循环的初始值。

    输出：
        Union[Tensor, number, str, bool, list, tuple, dict]，循环的最终结果，和 `init_val` 的类型相同。

    异常：
        - **TypeError** - `cond_func` 不是一个函数。
        - **TypeError** - `loop_func` 不是一个函数。
        - **ValueError** - `loop_func` 不能接受 `init_val` 作为参数或者返回值和 `init_val` 的类型不同。
