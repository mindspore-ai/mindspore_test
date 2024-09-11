mindspore.mint.nn.PReLU
=======================

.. py:class:: mindspore.mint.nn.PReLU(num_parameters=1, init=0.25, dtype=None)

    逐元素计算PReLU（PReLU Activation Operator）激活函数。

    公式定义为：

    .. math::

        PReLU(x_i)= \max(0, x_i) + w * \min(0, x_i),

    其中 :math:`x_i` 是输入的Tensor。

    这里 :math:`w` 是一个可学习的参数，默认初始值0.25。
    当带参数调用时每个通道上学习一个 :math:`w` 。如果不带参数调用时，则将在所有通道中共享单个参数 :math:`w` 。
    
    PReLU函数图：

    .. image:: ../images/PReLU2.png
        :align: center

    .. note::
        通道数是输入的第二个维度值。当输入的维度小于2时，则没有通道维度并且通道数等于1。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **num_parameters** (int) - 可训练参数 :math:`w` 的数量。虽然这个参数可以接受一个int作为输入，但只有两个值是合法的，值是1或输入Tensor `input` 的通道数。默认值： ``1`` 。
        - **init** (float) - 参数的初始值。默认值： ``0.25`` 。
        - **dtype** (mindspore.dtype，可选) - 参数的dtype。默认值： ``None`` 。支持的数据类型是{float16, float32, bfloat16}。

    输入：
        - **input** (Tensor) - PReLU的输入Tensor。

    输出：
        Tensor，数据类型和shape与 `input` 相同。
