mindspore.mint.linalg.norm
==========================

.. py:function:: mindspore.mint.linalg.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)

    返回给定Tensor的矩阵范数或向量范数。

    `ord` 为范数的计算模式。支持下列范数模式。

    =================   ================================== ==============================================
    `ord`                矩阵范数                               向量范数
    =================   ================================== ==============================================
    `None` (默认值)       Frobenius 范数                     `2`-范数 (参考最下方公式)
    `'fro'`              Frobenius 范数                      不支持
    `'nuc'`              Nuclear 范数                        不支持
    `inf`                :math:`max(sum(abs(x), dim=1))`    :math:`max(abs(x))`
    `-inf`               :math:`min(sum(abs(x), dim=1))`    :math:`min(abs(x))`
    `0`                  不支持                              :math:`sum(x != 0)`
    `1`                  :math:`max(sum(abs(x), dim=0))`    参考最下方公式
    `-1`                 :math:`min(sum(abs(x), dim=0))`    参考最下方公式
    `2`                  最大奇异值                           参考最下方公式
    `-2`                 最小奇异值                           参考最下方公式
    其余int或float值       不支持                             :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    =================   ================================== ==============================================

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **A** (Tensor) - shape为 :math:`(*, n)` 或者 :math:`(*, m, n)` 的Tensor。其中*是零个或多个batch维度。
        - **ord** (Union[int, float, inf, -inf, 'fro', 'nuc'], 可选) - 范数的计算模式。行为模式参考上表。默认值： ``None`` 。
        - **dim** (Union[int, Tuple(int)], 可选) - 计算向量范数或矩阵范数的维度。默认值： ``None`` 。

          - 当 `dim` 为int时，会按向量范数计算。
          - 当 `dim` 为一个二元组时，会按矩阵范数计算。
          - 当 `dim` 为 ``None`` 且 `ord` 为 ``None``，`A` 将会被展平为1D并计算向量的2-范数。
          - 当 `dim` 为 ``None`` 且 `ord` 不为 ``None``，`A` 必须为1D或者2D。

        - **keepdim** (bool) - 输出Tensor是否保留原有的维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果设置此参数，则会在执行之前将 `A` 转换为指定的类型，返回的Tensor类型也将为指定类型 `dtype`。默认值： ``None`` 。

    返回：
        Tensor，在指定维度 `dim` 上进行范数计算的结果，与输入 `A` 的数据类型相同。

    异常：
        - **ValueError** - `dim` 超出范围。
        - **TypeError** - `dim` 既不是int也不是由int组成的tuple。
        - **TypeError** - `A` 是一个向量并且 `ord` 是str类型。
        - **ValueError** - `A` 是一个矩阵并且 `ord` 不是有效的取值。
        - **ValueError** - `dim` 的两个元素在标准化过后取值相同。
        - **ValueError** - `dim` 的任意元素超出索引。

    .. note::
        动态shape、动态rank和可变输入不支持在 `图模式(mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下执行。