mindspore.ops.heaviside
========================

.. py:function:: mindspore.ops.heaviside(input, values)

    逐元素进行Heaviside阶跃函数运算。

    支持广播。

    .. math::
            \text { heaviside }(\text { input, values })=\left\{\begin{array}{ll}
            0, & \text { if input }<0 \\
            \text { values, } & \text { if input }=0 \\
            1, & \text { if input }>0
            \end{array}\right.

    参数：
        - **input** (Tensor) - 输入tensor。
        - **values** (Tensor) - `input` 中元素为0时填充的值。

    返回：
        Tensor