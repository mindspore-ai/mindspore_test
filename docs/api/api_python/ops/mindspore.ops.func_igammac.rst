mindspore.ops.igammac
=====================

.. py:function:: mindspore.ops.igammac(input, other)

    计算正则化的上层不完全伽马函数。
    将 `input` 记作 `a` ， `other` 记作 `x` ，则正则化的上层不完全伽马函数可以表示为：

    .. math::
        Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)

    其中，

    .. math::
        Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt

    表示上层不完全伽马函数。

    :math:`P(a, x)` 表示正则化的下层不完全伽马函数。   
 
    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型为float32或float64。
        - **other** (Tensor) - 输入Tensor，数据类型为float32或float64，与 `input` 保持一致。

    返回：
        Tensor，数据类型与 `input` 和 `other` 相同。

    异常：
        - **TypeError** - 如果 `input` 或者 `other` 不是Tensor。
        - **TypeError** - 如果 `other` 的数据类型不是float32或者float64。
        - **TypeError** - 如果 `other` 的数据类型与 `input` 不相同。
        - **ValueError** - 如果 `input` 不能广播成shape与 `other` 相同的Tensor。
