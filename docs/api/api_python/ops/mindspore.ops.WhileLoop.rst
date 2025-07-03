mindspore.ops.WhileLoop
=======================

.. py:class:: mindspore.ops.WhileLoop

    在编译阶段不进行循环展开的循环算子。
    WhileLoop算子的执行逻辑可以近似表示为如下代码:

    .. code-block:: python

        def WhileLoop(cond_func, loop_func, init_val):
            while(cond_func(init_val)):
                init_val = loop_func(init_val)
            return init_val

    当前WhileLoop算子存在以下语法限制：

    - 暂不支持 `loop_func` 为副作用函数，如：对Parameter、全局变量的修改等操作。
    - 暂不支持 `loop_func` 的返回值与初始值 `init_val` 的类型或形状不同。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **cond_func** (Function) - 循环的条件函数。
        - **loop_func** (Function) - 循环体函数，接受一个参数，并且返回值与输入参数的类型相同。
        - **init_val** (Union[Tensor, number, str, bool, list, tuple, dict]) - 循环的初始值。

    输出：
        Union[Tensor, number, str, bool, list, tuple, dict]，循环的最终结果，和 `init_val` 的类型和形状相同。

    异常：
        - **TypeError** - `cond_func` 不是一个函数。
        - **TypeError** - `loop_func` 不是一个函数。
        - **ValueError** - `loop_func` 不能接受 `init_val` 作为参数或者返回值和 `init_val` 的类型或形状不同。
