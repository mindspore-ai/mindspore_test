mindspore.mint.nn.ConvTranspose2d
=================================

.. py:class:: mindspore.mint.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros", dtype=None)

    将2D转置卷积运算应用于由多个输入平面组成的输入图像。

    这个模块可以看作是Conv2d相对于其输入的梯度。它也被称为分数跨步卷积或反卷积（尽管它不是实际的反卷积操作，其不是计算卷积的真逆函数）。

    所有参数中的 `kernel_size`， `stride`， `padding`， `output_padding` 可以为：

    - 单个整数 -- 在这种情况下，该值同时被用于H和W维度
    - 一个由两个整数组成的tuple -- 在这种情况下，第一个整数用于H维度，第二个整数用于W维度。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 在输入非连续场景下， `output_padding` 必须小于 `stride` 。
        - 在Atlas训练系列产品上，float32类型输入时，仅支持 `groups` 为1。

    参数：
        - **in_channels** (int) - 输入图像中的通道数。
        - **out_channels** (int) - 卷积生成的通道数。
        - **kernel_size** (Union[int, tuple(int)]) - 卷积核的大小。
        - **stride** (Union[int, tuple(int)], 可选) - 卷积的步长。默认值： ``1`` 。
        - **padding** (Union[int, tuple(int)], 可选) - :math:`dilation * (kernel\_size - 1) - padding` 零填充将添加到输入中每个维度的两侧。默认值： ``0`` 。
        - **output_padding** (Union[int, tuple(int)], 可选) - 在输出形状中每个维度的一侧增加额外的尺寸。 `output_padding` 的值必须小于 `stride` 或 `dilation` 。默认值： ``0`` 。
        - **groups** (int, 可选) - 从输入通道到输出通道的分块数。默认值： ``1`` 。
        - **bias** (bool, 可选) - 如果值为 ``True`` ， 则添加一个可学习的偏置至输出中。默认值： ``True`` 。
        - **dilation** (Union[int, tuple(int)], 可选) - 内核元素之间的间距。默认值： ``1`` 。
        - **padding_mode** (str, 可选) - 指定填充值的填充模式。目前仅支持 ``zeros``。默认值： ``zeros`` 。
        - **dtype** (mindspore.dtype, 可选) - 模型参数的类型。默认值： ``None``，此时模型参数类型为 ``mstype.float32`` 。

    可学习参数：
        - **weigh** (Parameter) - 模型的可学习权重，其shape为 :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}}, \text{kernel_size[0]}, \text{kernel_size[1]})`。
          其值从分布 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样得到，其中 :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel_size}[i]}` 。
        - **bias** (Parameter) - 模型的可学习偏置，其shape为 :math:`(\text{out_channels},)` 。
          如果 `bias` 为True，则其值从分布 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中采样得到，其中 :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel_size}[i]}`

    输入：
        - **input** (Tensor) - Tensor，其shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 或 :math:`(C_{in}, H_{in}, W_{in})` 。

    输出：
        Tensor， shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 或 :math:`(C_{out}, H_{out}, W_{out})` ，其中：

        .. math::
            H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                    \times (\text{kernel_size}[0] - 1) + \text{output_padding}[0] + 1
        .. math::
            W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) + \text{output_padding}[1] + 1
