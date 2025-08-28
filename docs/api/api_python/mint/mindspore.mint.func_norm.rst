mindspore.mint.norm
===================

.. py:function:: mindspore.mint.norm(input, p='fro', dim=None, keepdim=False, *, dtype=None)

    返回给定Tensor的矩阵范数或向量范数。

    `p` 为范数的计算模式。支持下列范数模式。

    =================   ================================== ==============================================
    `p`                  矩阵范数                               向量范数
    =================   ================================== ==============================================
    `'fro'`              Frobenius 范数                      不支持
    `'nuc'`              Nuclear 范数                        不支持
    其余int或float值       不支持                             :math:`sum(abs(x)^{p})^{(1 / p)}`
    =================   ================================== ==============================================

    参数：
        - **input** (Tensor) - shape为 :math:`(*)` 或者 :math:`(*, m, n)` 的Tensor，其中*是零个或多个batch维度。
        - **p** (Union[bool, int, float, inf, -inf, 'fro', 'nuc'], 可选) - 范数的计算模式。行为参考上表。默认值： ``'fro'`` 。
        - **dim** (Union[int, List(int), Tuple(int)], 可选) - 计算向量范数或矩阵范数的维度。默认值： ``None`` 。
        - **keepdim** (bool, 可选) - 输出Tensor是否保留原有的维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果设置此参数，则会在执行之前将 `input` 转换为指定的类型，返回的Tensor类型也将为指定类型 `dtype`。默认值： ``None`` 。

    返回：
        Tensor，在指定维度 `dim` 上进行范数计算的结果。

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **ValueError** - `dim` 超出范围。
        - **TypeError** - `dim` 既不是int也不是由int组成的tuple或list。
        - **ValueError** - `dim` 的两个元素在标准化过后取值相同。
        - **ValueError** - `dim` 的任意元素超出索引。

    .. note::
        动态shape、动态rank和可变输入不支持在 `图模式(mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下执行。
