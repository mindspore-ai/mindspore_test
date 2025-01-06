mindspore.mint.polar
====================

.. py:function:: mindspore.mint.polar(abs, angle)

    将极坐标转化为笛卡尔坐标。

    返回一个复数Tensor，其元素是由输入极坐标构造的笛卡尔坐标。其中极坐标由极径 `abs` 和极角 `angle` 给定。

    .. math::

        y_{i} =  abs_{i} * \cos(angle_{i}) + abs_{i} * \sin(angle_{i}) * j


    参数：
        - **abs** (Tensor, float) - 极径。其输入shape可以是任意维度，数据类型须为：float32。
        - **angle** (Tensor, float) - 极角。其shape与数据类型与 `abs` 一致。

    返回：
        Tensor，其shape与 `abs` 一致，dtype是complex64。


    异常：
        - **TypeError** - `abs` 或 `angle` 不是Tensor。
        - **TypeError** - 输入数据类型不是float32。
        - **TypeError** - `abs` 和 `angle` 数据类型不一致。
        - **ValueError** - `abs` 和 `angle` 的shape不一致。
