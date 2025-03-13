mindspore.ops.einsum
====================

.. py:function:: mindspore.ops.einsum(equation, *operands)

    基于爱因斯坦求和约定（Einsum）符号，沿指定维度计算输入tensor元素的乘积之和。

    .. note::
        - 现在支持子列表模式。例如，ops.einsum(op1, sublist1, op2, sublist2, ..., sublist_out)。在子列表模式中， `equation` 由子列表推导得到，Python的省略号和介于[0, 52)的整数list组成子列表。每个操作数后面都有一个子列表，并且最后有一个表示输出的子列表。
        - `equation` 只能包含字母、逗号、省略号和箭头。字母表示输入tensor维数，逗号表示单独的tensor，省略号表示忽略的tensor维数，箭头的左边表示输入tensor，右边表示期望输出的维度。

    参数：
        - **equation** (str) - 基于爱因斯坦求和约定的符号。
        - **operands** (Tensor) - 输入tensor。

    返回：
        Tensor
