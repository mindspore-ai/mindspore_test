mindspore.mint.nn.Conv1d
========================

.. py:class:: mindspore.mint.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None)

    一维卷积层。

    对输入Tensor计算一维卷积，通常输入的shape为 :math:`(N, C_{in}, L_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数， :math:`L` 为特征序列长度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{input}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作，
    :math:`weight` 为卷积核的值， :math:`input` 为输入的特征图。

    - :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    - :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    - :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]` ，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个\
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{input}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。
    卷积核shape为 :math:`(\text{kernel_size},)` ，其中 :math:`\text{kernel_size}` 是卷积核的长度。若考虑到输入输出通道以及groups，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size})` ，
    其中 `groups` 是分组卷积时在通道上分割输入 `input` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    参数：
        - **in_channels** (int) - Conv1d层输入Tensor的空间维度。
        - **out_channels** (int) - Conv1d层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 指定一维卷积核的长度。数据类型为整型或一个整型的tuple/list。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 一维卷积核的移动步长。数据类型为整型或者长度为1的整型tuple/list。默认值： ``1`` 。
        - **padding** (Union[int, tuple[int], list[int], str]，可选) - 输入的长度方向上填充的数量。数据类型为int或包含1个整数的tuple/list或string { ``"valid"`` ，  ``"same"`` } 。值应该要大于等于0。默认值： ``0`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在两侧，若为奇数，多余的填充量将补充在右侧。若设置该模式， `stride` 的值必须为1。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大长度，不能构成一个完整stride的额外的序列将被丢弃。

        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 卷积核膨胀尺寸。可以为单个int，或者由1个int组成的tuple/list。
          假设 :math:`dilation=(d)`, 则卷积核在长度方向间隔 :math:`d-1` 个元素进行采样。默认值： ``1`` 。
        - **groups** (int，可选) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `groups` 整除。如果组数等于 `in_channels` 和 `out_channels` ，这个一维卷积层也被称为一维深度卷积层。默认值： ``1`` 。
          需要满足以下约束：

          - :math:`(C_{in} \text{ % } \text{groups} == 0)`
          - :math:`(C_{out} \text{ % } \text{groups} == 0)`
          - :math:`(C_{out} >= \text{groups})`
          - :math:`(\text{weight[1]} = C_{in} / \text{groups})`

        - **bias** (bool，可选) - Conv1d层是否添加偏置参数。默认值： ``True`` 。

        - **padding_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"zeros"`` 、 ``"reflect"`` 或 ``"replicate"`` 。默认值： ``"zeros"`` 。

        - **dtype** (:class:`mindspore.dtype`，可选) - Parameters的dtype。默认值： ``None``， 使用 ``mstype.float32`` 。

    可变参数：
        - **weight** (Tensor) - 卷积层的权重，shape :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]})` 。
        - **bias** (Tensor) - 卷积层的偏置，shape :math:`(C_{out})` 。如果 `bias` 为False，则为None。

    输入：
        - **input** (Tensor) - Shape为 :math:`(N, C_{in}, L_{in})` 或者 :math:`(C_{in}, L_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, L_{out},)` 或者 :math:`(C_{out}, L_{out},)` 。

        padding为 ``"same"`` 时：

        .. math::
            \begin{array}{ll} \\
                L_{out} = \left \lceil{\frac{L_{in}}{\text{stride}}} \right \rceil \\
            \end{array}

        padding为 ``"valid"`` 时：

        .. math::
            \begin{array}{ll} \\
                L_{out} = \left \lfloor{\frac{L_{in} - \text{dilation} \times (\text{kernel_size} - 1) - 1}
                {\text{stride}}} \right \rfloor + 1 \\
            \end{array}

        padding为int或tuple/list时：

        .. math::
            \begin{array}{ll} \\
                L_{out} = \left \lfloor{\frac{L_{in} + 2 \times {padding} - \text{dilation} \times
                (\text{kernel_size} - 1) - 1}{\text{stride}}} \right \rfloor + 1 \\
            \end{array}

    异常：
        - **ValueError** - 输入特征图的大小与参数应满足输出公式，以确保输出特征图大小为正，否则会报错。
        - **RuntimeError** - Ascend上受不同型号NPU芯片上L1缓存大小限制，用例尺寸或Kernel Size过大。
        - **TypeError** - 如果 `in_channels` ， `out_channels` 或者 `groups` 不是整数。
        - **TypeError** - 如果 `kernel_size` ， `stride`， 或者 `dilation` 既不是int也不是tuple/list。
        - **ValueError** - 如果 `in_channels` ， `out_channels`， `kernel_size` ， `stride` 或者 `dilation` 小于1。
        - **ValueError** - 如果 `padding` 小于0。
        - **ValueError** - 如果 `padding` 是 ``"same"`` ， `stride` 不等于1。
        - **ValueError** - 输入参数不满足卷积输出公式。
        - **ValueError** - `kernel_size` 不能超过输入特征图的大小。
        - **ValueError** - `padding` 值不能导致计算区域超出输入大小。
