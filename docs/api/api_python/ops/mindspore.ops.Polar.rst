mindspore.ops.Polar
====================

.. py:class:: mindspore.ops.Polar

    将极坐标转化为笛卡尔坐标。

    返回一个复数Tensor，其元素是由输入极坐标构造的笛卡尔坐标。其中极坐标由极径 `abs` 和极角 `angle` 给定。

    更多细节请参考 :func:`mindspore.ops.polar`。

    .. math::

        y_{i} =  abs_{i} * \cos(angle_{i}) + abs_{i} * \sin(angle_{i}) * j

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **abs** (Tensor, float) - 极径，其输入shape可以是任意维度，其数据类型须为：float32。
        - **angle** (Tensor, float) - 极角，其shape和数据类型与 `abs` 一致。

    输出：
        Tensor，其shape与 `abs` 一致，dtype是complex64。
