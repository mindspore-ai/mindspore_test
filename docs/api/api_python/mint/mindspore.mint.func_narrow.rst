mindspore.mint.narrow
=====================

.. py:function:: mindspore.mint.narrow(input, dim, start, length) -> Tensor

    沿着指定的轴，指定起始位置获取指定长度的Tensor。

    参数：
        - **input** (Tensor) - 需要计算的Tensor。
        - **dim** (int) - 指定的轴。
        - **start** (Union[int, Tensor[int]]) - 指定起始位置。
        - **length** (int) - 指定长度。

    返回：
        Tensor，缩小后的Tensor。

    异常：
        - **ValueError** - `input` 的秩为0。
        - **ValueError** - `dim` 值超出范围[-input.ndim, input.ndim)。
        - **ValueError** - `start` 值超出范围[-input.shape[dim], input.shape[dim]]。
        - **ValueError** - `length` 值超出范围[0, input.shape[dim]-start]。
