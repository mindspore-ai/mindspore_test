mindspore.nn.MaxPool3d
=======================

.. py:class:: mindspore.nn.MaxPool3d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)

    在一个输入Tensor上应用3D最大池化运算，输入Tensor可看成是由一系列3D平面组成的。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` ，MaxPool3d输出的shape为 :math:`(D_{in}, H_{in}, W_{in})` 维度区域最大值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 `stride` 为 :math:`s = (s_0, s_1, s_2)`，公式如下。

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    .. note::
        Atlas 训练系列产品暂不支持此接口。

    参数：
        - **kernel_size** (Union[int, tuple[int]]，可选) - 指定池化核尺寸大小。如果为整数或单元素tuple，则该值同时代表池化核的深度、高和宽；如果为tuple且长度不为1，其值必须包含三个正整数值，分别表示池化核的深度、高和宽。取值必须为正整数。默认值： ``1`` 。
        - **stride** (Union[int, tuple[int]]，可选) - 池化操作的移动步长。如果为整数或单元素tuple，则该值同时代表池化核的深度、高和宽方向的移动步长；如果为tuple且长度不为1，其值必须包含三个正整数值，分别表示池化核的深度、高和宽的移动步长。取值必须为正整数。如果值为 ``None`` ，则使用默认值 `kernel_size`。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式。填充值为 ``0``。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的深度、高度和宽度维度进行填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周；若为奇数，多余的填充量将补充在前方、底部、右侧。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大深度、高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的深度、高度和宽度方向上填充的量由 `padding` 参数指定。如果设置了此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int], list[int])，可选) - 池化填充值。默认值： ``0`` 。 `padding` 只能是一个整数，或者包含一个或三个整数的tuple/list。若 `padding` 为一个整数或包含一个整数的tuple/list，则会分别在输入的前、后、上、下、左、右，六个方向进行 `padding` 次的填充；若 `padding` 为一个包含三个整数的tuple/list，则会在输入的前、后进行 `padding[0]` 次的填充，在输入的上、下进行 `padding[1]` 次的填充，在输入的左、右进行 `padding[2]` 次的填充。
        - **dilation** (Union(int, tuple[int])，可选) - 卷积核中各个元素之间的间隔大小，用于提升池化操作的感受野。如果为tuple，其值必须包含一个或三个整数。默认值： ``1`` 。
        - **return_indices** (bool，可选) - 若为True，则返回一个包含两个Tensor的Tuple，表示池化的计算结果以及生成max值的位置。否则，仅返回池化计算结果。默认值： ``False`` 。
        - **ceil_mode** (bool，可选) - 若为 ``True`` ，则使用ceil模式来计算输出shape；若为 ``False`` ，则使用floor模式来计算输出shape。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` 或者 :math:`(C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。

    输出：
        如果 `return_indices` 为 ``False`` ，则是shape为 :math:`(N_{out}, C_{out}, D_{out}, H_{out}, W_{out})` 或者 :math:`(C_{out}, D_{out}, H_{out}, W_{out})` 的Tensor。数据类型与 `x` 一致。
        如果 `return_indices` 为 ``True`` ，则是一个包含了两个Tensor的Tuple，表示maxpool的计算结果，以及生成max值的位置。

        - **output** (Tensor) - 最大池化结果。shape为 :math:`(N_{out}, C_{out}, D_{out}, H_{out}, W_{out})` 或者 :math:`(C_{out}, D_{out}, H_{out}, W_{out})` 的Tensor。数据类型与 `x` 一致。
        - **argmax** (Tensor) - 最大值对应的索引。数据类型为int64。

        其中，如果 `pad_mode` 为 ``"pad"`` 模式时，输出的shape计算公式如下：

        .. math::
            D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
            (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
            (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
            (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    异常：
        - **ValueError** - `x` 的shape长度不等于 4 或 5。
        - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 、 `dilation` 既不是int也不是tuple。
        - **ValueError** - `kernel_size` 或者 `stride` 小于1。
        - **ValueError** - `padding` 不为int也不是长度为3的tuple。
        - **ValueError** - `pad_mode` 不为 ``"pad"`` 模式时， `return_indices` 设为了 ``True`` 或者 `dilation` 不为1。
        - **ValueError** - `pad_mode` 不为 ``"pad"`` 模式时 `padding` 为非0。
