mindspore.mint.bernoulli
=========================

.. py:function:: mindspore.mint.bernoulli(input, *, generator=None)


    从伯努利分布中进行采样，并根据输入 `input` 中第 `i` 个元素给出的概率值将输出 `output` 中的第 `i` 元素随机设置为0或1。

    .. math::

        output_{i} \sim Bernoulli(p=input_{i})


    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 伯努利分布的输入张量，其中元素 `input_{i}` 代表对应输出元素 `output_{i}` 被设为 `1` 的概率，因此 `input` 中每个元素的数值范围都应当在 `[0, 1]` 之间。支持的数据类型： float16、float32、float64、bfloat16（仅Atlas A2训练系列产品支持）。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`, 可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。

    返回：
        - **output** (Tensor) - 输出张量，其shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 的数据类型不是float16、float32、float64、bfloat16之一。
        - **ValueError** - `input` 中任意一个元素的数值范围不在0到1之间。
