mindspore.mint.nn.functional.pixel_shuffle
==========================================

.. py:function:: mindspore.mint.nn.functional.pixel_shuffle(input, upscale_factor) -> Tensor

    根据上采样系数重排Tensor中的元素。

    将shape为 :math:`(*, C \times r^2, H, W)` 的Tensor重排shape为 :math:`(*, C, H \times r, W \times r)` 的Tensor，其中r是放大因子。

    这是步长为 :math:`1/r` 的子像素卷积。

    关于pixel_shuffle算法详细介绍，请参考 `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_ 。

    参数：
        - **input** (Tensor) - Tensor，shape为 :math:`(*, C \times r^2, H, W)` 。 `input` 的维度需要大于2，并且倒数第三维length可以被 `upscale_factor` 的平方整除。
        - **upscale_factor** (int) - 打乱输入Tensor的因子，是正整数。 `upscale_factor` 是上面提到的 :math:`r` 。

    返回：
        - **output** (Tensor) - Tensor，shape为 :math:`(*, C, H \times r, W \times r)` 。

    异常：
        - **ValueError** - `upscale_factor` 不是正整数。
        - **ValueError** - `input` 倒数第三维度的length不能被 `upscale_factor` 的平方整除。
        - **ValueError** - `input` 维度小于3。
