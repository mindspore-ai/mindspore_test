mindspore.dataset.vision.ResizedCrop
====================================

.. py:class:: mindspore.dataset.vision.ResizedCrop(top, left, height, width, size, interpolation=Inter.BILINEAR)

    裁剪输入图像的指定区域，并缩放到指定尺寸大小。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    参数：
        - **top** (int) - 裁剪区域左上角位置的纵坐标。
        - **left** (int) - 裁剪区域左上角位置的横坐标。
        - **height** (int) - 裁剪区域的高度。
        - **width** (int) - 裁剪区域的宽度。
        - **size** (Union[int, Sequence[int, int]]) - 图像的输出尺寸大小。

          - 若输入int，将调整图像的较短边长度为 `size` ，且保持图像的宽高比不变。
          - 若输入Sequence[int, int]，其输入格式需要是 (高度, 宽度) 。

        - **interpolation** (:class:`~.vision.Inter`, 可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.BILINEAR``。

    异常：
        - **TypeError** - 如果 `top` 不为int类型。
        - **ValueError** - 如果 `top` 为负数。
        - **TypeError** - 如果 `left` 不为int类型。
        - **ValueError** - 如果 `left` 为负数。
        - **TypeError** - 如果 `height` 不为int类型。
        - **ValueError** - 如果 `height` 不为正数。
        - **TypeError** - 如果 `width` 不为int类型。
        - **ValueError** - 如果 `width` 不为正数。
        - **TypeError** - 如果 `size` 不为int或Sequence[int, int]类型。
        - **ValueError** - 如果 `size` 不为正数。
        - **TypeError** - 如果 `interpolation` 不为 :class:`mindspore.dataset.vision.Inter` 类型。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        当执行设备是 Ascend 时，输入数据支持 `uint8` 和 `float32` 类型，输入数据的通道仅支持 1和3。输入数据的高度限制范围为[4, 32768]，宽度限制范围为[6, 32768]。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``"CPU"`` 和 ``"Ascend"`` 。默认值： ``"CPU"`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为[ ``"CPU"`` , ``"Ascend"`` ]。
