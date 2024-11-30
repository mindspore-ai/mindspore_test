mindspore.mint.nn.functional.conv3d
====================================

.. py:function:: mindspore.mint.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

    对输入Tensor计算三维卷积。通常，输入Tensor的shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数，:math:`D, H, W` 分别为特征图的深度、高度和宽度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作， 
    :math:`weight` 为卷积核的值， :math:`X` 为输入的特征图。

    - :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    - :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    - :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]` ，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{X}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。

    卷积核shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 :math:`\text{kernel_size[0]}` 、
    :math:`\text{kernel_size[1]}` 和 :math:`\text{kernel_size[2]}` 分别是卷积核的深度、高度和宽度。若考虑到输入输出通道以及 `groups` ，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，
    其中 `groups` 是分组卷积时在通道上分割输入 `x` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。
        - **weight** (Tensor) - shape为 :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`  ，则卷积核的大小为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` 。
        - **bias** (Tensor，可选) - 偏置Tensor，shape为 :math:`(C_{out})` 的Tensor。如果 `bias` 是None，将不会添加偏置。默认值： ``None`` 。
        - **stride** (Union[int, tuple[int]]，可选) - 卷积核移动的步长，可以为单个int或三个int组成的tuple。一个int表示在深度、高度和宽度方向的移动步长均为该值。三个int组成的tuple分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **padding** (Union(int, tuple[int], str)，可选) - 输入 `x` 两侧的隐式填充。可以是字符串、一个整数或包含3个整数的元组/列表。如果 `padding` 是一个字符串，则可选值为 `same` 、 `valid` 。

          - ``"same"``：采用完成方式。输出的高度和宽度将等于输入 `x` 除以步幅。填充将尽可能在顶部和底部、左侧和右侧均匀计算。否则，最后一个额外的填充将从底部和右侧计算。如果设置了此模式，则 `padding` 必须为0。
          - ``"valid"``：采用丢弃的方式。输出的可能最大高度和宽度将在没有填充的情况下返回。多余的像素将被丢弃。如果设置了此模式，则 `padding` 必须为0。
          
          如果 `padding` 是一个整数，则top、bottom、left和right的padding是相同的，等于padding。
          如果 `padding` 是一个包含3个整数的元组/列表，则head、tail、top、bottom、left和right的填充分别等于pad[0]、pad[0]、pad[1]、pad[2]和pad[2]。默认值： `0` 。

        - **dilation** (Union[int, tuple[int]]，可选) - 控制内核点之间的空间。默认值： ``1`` 。
        - **groups** (int，可选) - 将 `input` 拆分的组数。默认值： ``1`` 。

    返回：
        Tensor。
        
    异常：
        - **TypeError** -  `stride` 、 `padding` 或 `dilation` 既不是int也不是tuple。
        - **TypeError** -  `groups` 不是int。
        - **TypeError** -  `bias` 不是Tensor。
