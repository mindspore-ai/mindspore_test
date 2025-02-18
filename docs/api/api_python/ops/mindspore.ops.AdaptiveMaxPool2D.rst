mindspore.ops.AdaptiveMaxPool2D
===============================

.. py:class:: mindspore.ops.AdaptiveMaxPool2D

    对一个多平面输入信号执行二维自适应最大值池化。

    更多参考详见 :func:`mindspore.ops.adaptive_max_pool2d`。

    输入：
        - **input** (Tensor) - `AdaptiveMaxPool2D` 的输入，为三维或四维的Tensor，数据类型为float16、float32或者float64。
        - **output_size** (tuple) - 输出特征图的size。 `output_size` 可以为二元tuple表示 :math:`(H, W)`。:math:`H` 和 :math:`W` 必须是int。

    输出：
        Tensor，数据类型与 `input` 相同。
