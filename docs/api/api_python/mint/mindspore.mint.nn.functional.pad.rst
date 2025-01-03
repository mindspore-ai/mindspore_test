mindspore.mint.nn.functional.pad
================================
.. py:function:: mindspore.mint.nn.functional.pad(input, pad, mode='constant', value=None)

    根据参数 `pad` 对输入进行填充。

    .. warning::
        `circular` 模式性能较差，不推荐使用。

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(N, *)`， :math:`*` 代表任意附加维度。
        - **pad** (Union[tuple[int], list[int], Tensor]) - pad的填充位置。
          :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` 维度的 `input` 将会被填充。可根据以下示例以此类推：

          - 示例：若只需要填充输入tensor的最后一个维度，则 `pad` 的填充方式为 :math:`(\text{padding_left}, \text{padding_right})`;
          - 示例：若只需要填充输入tensor的最后两个维度，则 `pad` 的填充方式为 :math:`(\text{padding_left}, \text{padding_right}, \text{padding_top}, \text{padding_bottom})`;
          - 示例：若只需要填充输入tensor的最后三个维度，则 `pad` 的填充方式为 :math:`(\text{padding_left}, \text{padding_right}, \text{padding_top}, \text{padding_bottom}, \text{padding_front}, \text{padding_back})`;

        - **mode** (str，可选) - pad的填充模式，可选择 ``'constant'`` 、 ``'reflect'`` 、 ``'replicate'`` 或者 ``'circular'`` 。默认值： ``'constant'`` 。

          - 对于 ``'constant'`` 模式，请参考 :class:`mindspore.nn.ConstantPad1d` 作为示例来理解这个填充模式，并将这个模式扩展到n维。
          - 对于 ``'reflect'`` 模式，请参考 :class:`mindspore.nn.ReflectionPad1d` 作为示例来理解这个填充模式，reflect模式用于填充四维或五维输入的最后三个维度、三维或四维输入的最后两个维度，或者二维或三维输入的最后一个维度。
          - 对于 ``'replicate'`` 模式，请参考 :class:`mindspore.nn.ReplicationPad1d` 作为示例来理解这个填充模式，replicate模式用于填充四维或五维输入的最后三个维度、三维或四维输入的最后两个维度，或者二维或三维输入的最后一个维度。
          - 对于 ``'circular'`` 模式，circular模式用于将图像的像素从一侧循环地填充到另一侧。例如，右侧的像素将被替换为左侧的像素，底部的像素将被替换为顶部的像素。circular模式用于填充四维或五维输入的最后三个维度、三维或四维输入的最后两个维度，或者二维或三维输入的最后一个维度。

        - **value** (Union[int, float, None]，可选) - 仅在 ``'constant'`` 模式下生效，设置在 ``'constant'`` 模式下的填充值，如果值为 ``None`` ，则会使用0作为默认填充值。默认值： ``None`` 。

    返回：
        填充后的Tensor。

    异常：
        - **TypeError** - `pad` 不是全为int的tuple或者list。
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `pad` 的长度不为偶数。
        - **ValueError** - `pad` 的长度大于6。
        - **ValueError** - `mode` 不为 ``'constant'`` 并且 `value` 既不是 ``None`` 也不是0。

