mindspore.Tensor.bernoulli\_
============================

.. py:method:: mindspore.Tensor.bernoulli_(p=0.5, *, generator=None)

    用来自Bernoulli(p)的独立样本填充输入的每个元素。

    参数：
        - **p** (Union[number.Number, Tensor], 可选) - `p` 应为标量或张量，其中包含用于生成二进制随机数的概率，取值范围为 ``0`` 到 ``1`` 。如果是张量，则 `p` 必须为浮点型。默认值： ``0.5`` 。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`, 可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。

    返回：
        返回输入tensor。
