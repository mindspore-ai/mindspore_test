mindspore.Tensor.div
====================

.. py:method:: mindspore.Tensor.div(other, *, rounding_mode=None)

    逐元素计算当前Tensor除以输入的另一个Tensor的商。

    .. math::
        out_{i} = input_{i} / other_{i}

    .. note::
        - 当两个输入具有不同的shape时，它们的shape必须要能广播为一个共同的shape。
        - 两个输入不能同时为bool类型。[True, Tensor(True, bool\_), Tensor(np.array([True]), bool\_)]等都为bool类型。
        - 两个输入遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **other** (Union[Tensor, Number, bool]) - 另一个输入，可以是数值型或bool，也可以是数据类型为数值型或bool的Tensor。

    关键字参数：
        - **rounding_mode** (str, 可选) - 应用于结果的舍入类型。三种类型被定义为 ``None`` 、 ``"floor"`` 和 ``"trunc"`` 。默认值： ``None`` 。

          - **None**：默认行为。相当于Python中的 `true division` 或NumPy中的 `true_divide` 。
          - **"floor"**：将除法的结果向下舍入。相当于Python中的 `floor division` 或NumPy中的 `floor_divide` 。
          - **"trunc"**：将除法的结果舍入到零。相当于C语言风格的整数除法。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或位数更多的。

    异常：
        - **TypeError** - 如果 `self` 和 `other` 不是以下之一：Tensor、Number、bool。
        - **ValueError** - 如果 `rounding_mode` 不是以下之一： ``None`` 、 ``"floor"`` 、 ``"trunc"`` 。
