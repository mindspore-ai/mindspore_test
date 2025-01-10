mindspore.Tensor.add\_
======================

.. py:method:: mindspore.Tensor.add_(other)

    `self` 和 `other` 逐元素相加， `self` 将进行原地更新。

    .. math::

        out_{i} = self_{i} + other_{i}

    .. note::
        - 当 `self` 和 `other` 具有不同的shape时，它们的shape必须要能广播为一个共同的shape。
        - `self` 和 `other` 不能同时为bool类型。[True, Tensor(True, bool\_), Tensor(np.array([True]), bool\_)]等都为bool类型。
        - `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - `self` 的维度应大于或等于1。
        - CPU、GPU的所有模式，以及Atlas训练系列产品的 `图模式(mode=mindspore.GRAPH_MODE) <https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/static_graph.html>`_ 尚不支持原地更新， `self` 将不会进行原地更新。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - `other` 是一个number.Number、bool值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。

    返回：
        Tensor，shape与输入 `self`、 `other` 广播后的shape相同，数据类型为两个输入中精度较高的类型。

    异常：
        - **TypeError** - `other` 不是Tensor、number.Number或bool。

    .. py:method:: mindspore.Tensor.add_(other, *, alpha=1)
        :noindex:

    对 `other` 缩放后与 `self` 相加， `self` 将进行原地更新。

    .. math::

        out_{i} = self_{i} + alpha \times other_{i}

    .. note::
        - 当 `self` 和 `other` 的shape不同时，
          它们必须能够广播到一个共同的shape。
        - `self`、 `other` 和 `alpha` 遵守隐式类型转换规则以使数据类型\
          保持一致。
        - CPU、GPU的所有模式，以及Atlas训练系列产品的 `图模式(mode=mindspore.GRAPH_MODE) <https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/static_graph.html>`_ 尚不支持原地更新， `self` 将不会进行原地更新。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - `other` 是一个 number.Number、一个 bool 或一个数据类型为
          `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或
          `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。

    关键字参数：
        - **alpha** (number.Number，可选) - 应用于 `other` 的缩放因子，默认值为 ``1``。

    返回：
        Tensor，其shape与 `self`、 `other` 广播后的shape相同，
        数据类型是 `self`、 `other` 和 alpha 中精度更高或位数更多的类型。

    异常：
        - **TypeError** - 如果 `other` 或 `alpha` 不是以下之一：Tensor、number.Number、bool。
        - **TypeError** - 如果 `alpha` 是 float 类型，但是 `self`、 `other` 不是 float 类型。
        - **TypeError** - 如果 `alpha` 是 bool 类型，但是 `self`、 `other` 不是 bool 类型。
