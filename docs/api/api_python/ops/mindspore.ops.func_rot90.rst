mindspore.ops.rot90
=======================

.. py:function:: mindspore.ops.rot90(input, k, dims)

    沿指定维度的平面内将n-D tensor旋转90度。
    如果k>0，旋转方向是从第一轴朝向第二轴，如果k<0，旋转方向从第二轴朝向第一轴。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **k** (int) - 旋转的次数。
        - **dims** (Union[list(int), tuple(int)]) - 指定维度。

    返回：
        Tensor
