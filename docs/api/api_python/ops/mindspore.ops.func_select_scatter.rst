mindspore.ops.select_scatter
============================

.. py:function:: mindspore.ops.select_scatter(input, src, axis, index)

    将 `src` 中的值散布到 `input` 指定轴 `axis` 的指定位置 `index` 上。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **src** (Tensor) - 源tensor。
        - **axis** (int) - 指定轴。
        - **index** (int) - 在指定轴上散布的位置。

    返回：
        Tensor
