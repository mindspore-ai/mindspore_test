mindspore.Tensor.allclose
=========================

.. py:method:: mindspore.Tensor.allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

    返回一个布尔型标量，表示 `self` 的每个元素均与 `other` 的对应元素在给定容忍度内“接近”。其中“接近”的数学公式为：

    .. math::
        |self-other| ≤ atol + rtol * |other|

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **other** (Tensor) - 对比的Tensor，数据类型必须与 `self` 相同。
        - **rtol** (Union[float, int, bool], 可选) - 相对容忍度。默认值： ``1e-05``。
        - **atol** (Union[float, int, bool], 可选) - 绝对容忍度。默认值： ``1e-08``。
        - **equal_nan** (bool, 可选) - 若为True，则两个NaN被视为相同。默认值： ``False`` 。

    返回：
        布尔型标量。

    异常：
        - **TypeError** - `self` 和 `other` 中的任何一个不是Tensor。
        - **TypeError** - `self` 和 `other` 的数据类型不在支持的类型列表中。
        - **TypeError** - `atol` 和 `rtol` 中的任何一个不是float、int或bool。
        - **TypeError** - `equal_nan` 不是bool。
        - **TypeError** - `self` 和 `other` 的数据类型不同。
        - **ValueError** - `self` 和 `other` 无法广播。