mindspore.ops.aminmax
======================

.. py:function:: mindspore.ops.aminmax(input, *, axis=0, keepdims=False)

    返回tensor在指定轴上的最小值和最大值。

    参数：
        - **input** (Tensor) - 输入tensor。

    关键字参数：
        - **axis** (int，可选) - 指定计算的轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``0`` 。
        - **keepdims** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        两个tensor组成的tuple(min, max)。
