mindspore.mint.nn.UpsamplingBilinear2d
======================================

.. py:class:: mindspore.mint.nn.UpsamplingBilinear2d(size=None, scale_factor=None)

    对四维输入Tensor执行二线性插值上采样。

    此运算符使用指定的 `size` 或 `scale_factor` 缩放因子放大输入体积，过程使用二线性上调算法。

    .. note::
        必须指定 `size` 或 `scale_factor` 中的一个值，并且不能同时指定两者。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **size** (Union[tuple[int], list[int]], 可选) - 指定输出体积大小的元组或int列表。默认值： ``None`` 。
        - **scale_factor** (Union[tuple[float], list[float]], 可选) - 指定上采样因子的元组或float列表。默认值： ``None`` 。

    输入：
        - **input** (Tensor) - Shape为 :math:`(N, C, H_{in}, W_{in})` 的四维Tensor。支持的数据类型：[float16, float32, float64]。

    输出：
        上采样输出。其shape :math:`(N, C, H_{out}, W_{out})` ，数据类型与 `input` 相同。

    异常：
        - **TypeError** - 当 `size` 不是 ``None`` 并且 `size` 不是list[int]或tuple[int]。
        - **TypeError** - 当 `scale_factor` 不是 ``None`` 并且 `scale_factor` 不是list[float]或tuple[float]。
        - **TypeError** - `input` 的数据类型不是float16、float32或float64。
        - **ValueError** - `size` 不为 ``None`` 时含有负数或0。
        - **ValueError** - `scale_factor` 不为 ``None`` 时含有负数或0。
        - **ValueError** - `input` 维度不为4。
        - **ValueError** - `scale_factor` 和 `size` 同时被指定或都不被指定。
        - **ValueError** - `scale_factor` 被指定时其含有的元素个数不为2。
        - **ValueError** - `size` 被指定时其含有的元素个数不为2。
