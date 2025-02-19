mindspore.Tensor.nan_to_num
============================

.. py:method:: mindspore.Tensor.nan_to_num(nan=None, posinf=None, neginf=None)

    将 `self` 中的 `NaN` 、正无穷大和负无穷大值分别替换为 `nan` 、`posinf` 和 `neginf` 指定的值。

    .. warning::
        对于Ascend，仅支持Atlas A2训练系列产品。
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **nan** (number，可选) - 替换 `NaN` 的值。默认值为 ``None`` 。
        - **posinf** (number，可选) - 如果是一个数字，则为替换正无穷的值。如果为 ``None`` ，则将正无穷替换为 `self` 类型支持的上限。默认值为 ``None`` 。
        - **neginf** (number，可选) - 如果是一个数字，则为替换负无穷的值。如果为 ``None`` ，则将负无穷替换为 `self` 类型支持的下限。默认值为 ``None`` 。

    返回：
        Tensor，数据shape和类型与 `self` 相同。

