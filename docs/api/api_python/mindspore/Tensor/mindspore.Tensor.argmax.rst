mindspore.Tensor.argmax
=======================

.. py:method:: mindspore.Tensor.argmax(axis=None, keepdims=False)

    返回 `self` 在指定轴上的最大值索引。

    参数：
        - **axis** (Union[int, None]，可选) - 指定计算轴。如果是 ``None`` ，将会返回扁平化Tensor在指定轴上的最大值索引。不能超过 `self` 的维度。默认值： ``None`` 。
        - **keepdims** (bool，可选) - 输出Tensor是否保留指定轴。如果 `axis` 是 ``None`` ，忽略该选项。默认值： ``False`` 。

    返回：
        Tensor，输出为 `self` 在指定轴上最大值的索引。

    异常：
        - **TypeError** - 如果入参 `keepdims` 的类型不是布尔值。
        - **ValueError** - 如果入参 `axis` 的设定值超出了范围。

    .. py:method:: mindspore.Tensor.argmax(dim=None, keepdim=False)
        :noindex:

    返回 `self` 在指定轴上的最大值索引。

    参数：
        - **dim** (Union[int, None]，可选) - 指定计算轴。如果是 ``None`` ，将会返回扁平化Tensor在指定轴上的最大值索引。不能超过input的维度。默认值： ``None`` 。
        - **keepdim** (bool，可选) - 输出Tensor是否保留指定轴。如果 `dim` 是 ``None`` ，忽略该选项。默认值： ``False`` 。

    返回：
        Tensor，输出为 `self` 在指定轴上的最大值索引。

    异常：
        - **TypeError** - 如果入参 `keepdim` 的类型不是布尔值。
        - **ValueError** - 如果入参 `dim` 的设定值超出了范围。
