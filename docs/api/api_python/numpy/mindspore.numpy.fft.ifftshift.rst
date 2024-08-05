mindspore.numpy.fft.ifftshift
=============================

.. py:function:: mindspore.numpy.fft.ifftshift(x, axes=None)

    `fftshift` 的逆运算。

    参见 :func:`mindspore.ops.ifftshift` 获取更多详情。 不同之处在于 `x` 对应 `input` , `axes` 对应 `dim` 。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **axes** (Union[int, list(int), tuple(int)], 可选) - 操作所沿的轴。 默认值： ``None`` ，表示对所有轴进行操作。

    返回：
        Tensor，进行移动后的Tensor，与 `x` 的shape和dtype相同。