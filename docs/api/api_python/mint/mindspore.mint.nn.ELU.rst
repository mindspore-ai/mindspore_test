mindspore.mint.nn.ELU
=====================

.. py:class:: mindspore.mint.nn.ELU(alpha=1.0)

    指数线性单元激活函数(Exponential Linear Unit activation function)。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::
        ELU_{i} =
        \begin{cases}
        x_i, &\text{if } x_i \geq 0; \cr
        \alpha * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    其中，:math:`x_i` 表示输入的元素，:math:`\alpha` 表示 `alpha` 参数， `alpha` 决定ELU的平滑度。

    ELU函数图：

    .. image:: ../images/ELU.png
        :align: center

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **alpha** (float, 可选) - ELU的alpha值，数据类型为float。默认值： ``1.0`` 。

    输入：
        - **input** (Tensor) - 用于计算ELU的任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `alpha` 不是float。
