mindspore.mint.argmax
=====================

.. py:function:: mindspore.mint.argmax(input) -> Tensor

    返回输入Tensor最大值索引。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor。

    .. py:function:: mindspore.mint.argmax(input, dim, keepdim=False) -> Tensor
        :noindex:

    返回输入Tensor在指定轴上的最大值索引。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (int) - 指定计算轴。
        - **keepdim** (bool，可选) - 输出Tensor是否保留指定轴。默认值： ``False`` 。

    返回：
        Tensor，输出为指定轴上输入Tensor最大值的索引。

    异常：
        - **TypeError** - 如果 `keepdim` 的类型不是bool值。
        - **ValueError** - 如果 `dim` 的设定值超出了范围。
