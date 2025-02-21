mindspore.mint.flip
===================

.. py:function:: mindspore.mint.flip(input, dims)

    沿给定轴翻转Tensor中元素的顺序。

    保留Tensor的shape，但是元素将重新排序。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dims** (Union[list[int], tuple[int]]) - 需要翻转的一个轴或多个轴。在tuple中指定的所有轴上执行翻转操作，如果 `dims` 是一个包含负数的整数tuple，则该轴为按倒序计数的轴位置。

    返回：
        返回沿给定轴翻转计算结果的Tensor。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `dims` 为None。
        - **ValueError** - `dims` 不为int组成的list或tuple。
