mindspore.mint.var_mean
=======================

.. py:function:: mindspore.mint.var_mean(input, dim=None, *, correction=1, keepdim=False)

    默认情况下，返回Tensor各维度上的方差和均值。如果 dim 是维度列表，则计算对应维度的方差和均值。

    方差 (:math:`\sigma ^2`) 计算如下：

    .. math::
        \sigma ^2 = \frac{1}{N - \delta N} \sum_{j=0}^{N-1} \left(self_{ij} - \overline{x_{i}}\right)^{2}

    其中 :math:`x` 表示用来计算方差的样本集， :math:`\bar{x}` 表示样本的均值， :math:`N` 表示样本的数量， :math:`\delta N` 为 `correction` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。支持的数据类型：float16、float32。
        - **dim** (Union[int, tuple(int), list(int)], 可选) - 指定计算方差和平均值的维度。默认值： ``None``。

    关键字参数：
        - **correction** (int, 可选) - 样本大小和样本自由度之间的差异。默认采用贝塞尔校正。默认值： ``1`` 。
        - **keepdim** (bool, 可选) - 是否保留输出Tensor的维度。如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    返回：
        包含方差和均值的tuple。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `dim` 不是以下数据类型之一：int、tuple、list或Tensor。
        - **TypeError** - `keepdim` 不是bool类型。
        - **ValueError** - `dim` 超出范围。