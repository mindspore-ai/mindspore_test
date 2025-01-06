mindspore.mint.nn.Conv3d
=============================

.. py:class:: mindspore.mint.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None)

    三维卷积层。

    对输入Tensor计算三维卷积。通常，输入Tensor的shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数，:math:`D, H, W` 分别为特征图的深度、高度和宽度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作， 
    :math:`weight` 为卷积核的值， :math:`X` 为输入的特征图。

    - :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    - :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    - :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]`，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个\
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{X}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。

    卷积核shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 :math:`\text{kernel_size[0]}` 、
    :math:`\text{kernel_size[1]}` 和 :math:`\text{kernel_size[2]}` 分别是卷积核的深度、高度和宽度。若考虑到输入输出通道以及 `group` ，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，
    其中 `group` 是分组卷积时在通道上分割输入 `x` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    参数的约束细节，请参考 :func:`mindspore.mint.nn.functional.conv3d` 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **in_channels** (int) - Conv3d层输入Tensor的空间维度。
        - **out_channels** (int) - Conv3d层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的高度和宽度。数据类型是整数或两个整数的元组。整数表示卷积核的高度和宽度。两个整数的元组分别表示卷积核的高度和宽度。
        - **stride** (Union[int, tuple[int]]，可选) - 三维卷积核的移动步长。数据类型为整型或三个整型的tuple。一个整数表示在深度、高度和宽度方向的移动步长均为该值。三个整数的tuple分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **padding** (Union[int, tuple[int], str]，可选) - 输入的深度、高度和宽度方向上的填充数。数据类型是整数或字符串{ `valid` ， `same` }或三个整数的元组。该值应大于或等于0。默认值： ``0`` 。

          - ``"same"``：在输入的边缘加上衬垫，这样当 `stride` 设置为“1”时，输入和输出的形状是相同的。填充量由运算符内部计算。如果填充量是偶数，则均匀分布在输入周围，如果填充量为奇数，则多余的填充量会流向右侧/底部。
            如果设置了此模式，则 `padding` 必须为0。
          - ``"valid"``：输入没有填充，输出返回最大可能的高度和宽度。无法完成整个步幅的额外像素将被丢弃。如果设置了此模式，则 `padding` 必须为0。

        - **padding_mode** (str，可选) - 使用填充值0指定填充模式。它可以设置为： ``"zeros"`` 、 ``"reflect"`` 、 ``"circular"`` 或 ``"replicate"`` 。默认值： ``"zeros"`` 。
        - **dilation** (Union[int, tuple[int]]，可选) - 控制内核点之间的空间。默认值： ``1`` 。
        - **groups** (int，可选) - 拆分过滤成组， `in_channel` 和 `out_channels` 必须能被 `groups` 整除。如果组等于 `in_channels` 和 `out_channels` 。默认值： `1` 。
        - **bias** (bool，可选) - Conv3d层是否具有偏置参数。默认值： `True` 。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前，CPU和GPU平台上输入数据类型支持float16和float32，Ascend平台上输入数据类型只支持float16。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

        pad_mode为 ``"same"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}

        pad_mode为 ``"valid"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - `in_channels` 、 `out_channels` 或 `group` 不是int。
        - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 或 `dilation` 既不是int也不是tuple。
        - **ValueError** - `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
