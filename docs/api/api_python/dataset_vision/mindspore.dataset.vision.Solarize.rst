mindspore.dataset.vision.Solarize
=================================

.. py:class:: mindspore.dataset.vision.Solarize(threshold)

    通过反转阈值内的所有像素值，对输入图像进行曝光。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    参数：
        - **threshold** (Union[float, Sequence[float, float]]) - 反转的像素阈值范围，应该以（min，max）的格式提供，
          其中min和max是[0，255]范围内的整数，并且min <= max，那么属于[min, max]这个区间的像素值会被反转。如果只提供一个值或min = max，则反转大于等与min（max）的所有像素值。

    异常：
        - **TypeError** - 如果 `threshold` 不为float或Sequence[float, float]类型。
        - **ValueError** - 如果 `threshold` 的取值不在[0, 255]范围内。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        当执行设备是 Ascend 时，输入数据仅支持 `uint8` 类型，输入数据的通道仅支持1和3。输入数据的高度限制范围为[4, 8192]，宽度限制范围为[6, 4096]。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``"CPU"`` 和 ``"Ascend"`` 。默认值： ``"CPU"`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为[ ``"CPU"`` , ``"Ascend"`` ]。
