mindspore.mint.nn.AdaptiveAvgPool2d
====================================

.. py:class:: mindspore.mint.nn.AdaptiveAvgPool2d(output_size)

    对由多个输入平面组成的输入信号，运用2D自适应平均池化。

    对于任何输入大小，输出大小均为 :math:`H x W` 。
    输出特征的数量等于输入平面的数量。

    参数：
        - **output_size** (Union(int, tuple[int])) - :math:`H x W` 形式的图像的目标输出尺寸。
          可以是tuple :math:`（H，W）` ，也可以是正方形图像 :math:`H x H` 的单个 :math:`H` 。
          :math:`H` 和 :math:`W` 可以是 ``int`` 或 ``None`` ，这意味着大小将与输入相同。

    输入：
        - **input** (Tensor) - 输入特征的shape为 :math:`(N, C, *)`，其中 :math:`*` 任意数量的附加维度。
